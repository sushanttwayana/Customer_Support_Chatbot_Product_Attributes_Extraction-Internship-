from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase

# Load environment variables
load_dotenv()

load_dotenv()
# print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI(title="SQL Chatbot API")

# Database connection parameters
username = 'root'
password = 'root'
host = '127.0.0.1'
db_name = 'nepa_wholesale'
encoded_password = quote_plus(password)
mysql_uri = f"mysql+pymysql://{username}:{encoded_password}@{host}/{db_name}"

# Global variables for database and tools
db = None
tools = []

# Initialize database connection
def initialize_database():
    global db, tools
    try:
        db = SQLDatabase.from_uri(mysql_uri, sample_rows_in_table_info=2)
        print("Database connected successfully.")
        
        # Define tools after successful database connection
        @tool
        def list_tables(dummy_input: str = "") -> str:
            """Get a list of all available tables in the database."""
            return str(db.get_table_names())

        @tool
        def describe_table(table_name: str) -> str:
            """Get detailed information about a specific table including its columns and sample data."""
            return str(db.get_table_info([table_name]))

        @tool
        def run_sql_query(query: str) -> str:
            """Execute a SQL query against the database and return the results."""
            try:
                result = db.run(query)
                return str(result)
            except Exception as e:
                return f"Error executing query: {str(e)}"
        
        tools = [list_tables, describe_table, run_sql_query]
        return True
        
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

# Initialize database on startup
database_connected = initialize_database()

# Few-shot examples for semantic similarity search
examples = [
    {
        "question": "How many products of fronto are there in tobaccos?",
        "query": "SELECT COUNT(*) FROM tobaccos_category WHERE Brand LIKE '%FRONTO%';"
    },
    {
        "question": "How many products are there in tobaccos with flavor mint?",
        "query": "SELECT COUNT(Product_ID) FROM tobaccos_category WHERE Flavor LIKE '%mint%';"
    },
    {
        "question": "List the top 5 disposable products with the highest puff count.",
        "query": "SELECT Display_Name, Puff_count FROM disposable_category ORDER BY Puff_count DESC LIMIT 5;"
    },
    {
        "question": "Which flavors of cigars have 'Raspberry' in them?",
        "query": "SELECT Flavor FROM cigars_category WHERE Flavor LIKE '%Raspberry%' LIMIT 5;"
    },
    {
        "question": "How many unique flavors are offered in disposable products with 5000 puffs?",
        "query": "SELECT COUNT(DISTINCT Flavor) FROM disposable_category WHERE Puff_count = 5000;"
    },
    {
        "question": "I'm looking for cigars from the brand AL CAPONE. What do you have?",
        "query": "SELECT Display_Name, Flavor FROM cigars_category WHERE Brand LIKE '%AL CAPONE%' LIMIT 5;"
    },
    {
        "question": "List disposable products with nicotine strength of 2%.",
        "query": "SELECT Display_Name, Nicotine_strength FROM disposable_category WHERE Nicotine_strength LIKE '%2%%' LIMIT 5;"
    },
    {
        "question": "How many products are there in the DEATH ROW DISPOSABLE sub-category with 2% nicotine and 5 pack count?",
        "query": "SELECT COUNT(*) FROM disposable_category WHERE Product_Sub_Category LIKE '%DEATH ROW DISPOSABLE%' AND Nicotine_strength LIKE '%2%%' AND Pack_count = 5;"
    },
    {
        "question": "Which brands offer disposable products with puff count equal to 5000?",
        "query": "SELECT DISTINCT Brand FROM disposable_category WHERE Puff_count = 5000 LIMIT 5;"
    },
    {
        "question": "How many different cigar flavors do you offer?",
        "query": "SELECT COUNT(DISTINCT Flavor) FROM cigars_category;"
    },
    {
        "question": "Show me all tobacco products priced below 500.",
        "query": "SELECT Product_ID, Display_Name, Price FROM tobaccos_category WHERE Price < 500 LIMIT 10;"
    },
    {
        "question": "What is the average puff count for disposable products?",
        "query": "SELECT AVG(Puff_count) FROM disposable_category;"
    },
    {
        "question": "List all products in cigars_category that were added after January 1, 2023.",
        "query": "SELECT Display_Name, Added_Date FROM cigars_category WHERE Added_Date > '2023-01-01' LIMIT 10;"
    },
    {
        "question": "How many products have 'Menthol' in their flavor and cost more than 1000?",
        "query": "SELECT COUNT(*) FROM tobaccos_category WHERE Flavor LIKE '%Menthol%' AND Price > 1000;"
    },
    {
        "question": "Get the top 3 brands with the most products in disposable_category.",
        "query": "SELECT Brand, COUNT(*) AS product_count FROM disposable_category GROUP BY Brand ORDER BY product_count DESC LIMIT 3;"
    }
]

# Initialize embedding model and vectorstore only if database is connected
vectorstore = None
example_selector = None

if database_connected:
    try:
        # Embedding model & vectorstore for few-shot example selection
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Explicit CPU usage
        )

        vectorstore = Chroma(
            embedding_function=embedding_model,
            collection_name="example_collection",
            persist_directory="./chroma_db"
        )

        # Check if vectorstore is empty, then add examples
        try:
            existing_docs = vectorstore.get()
            if not existing_docs['ids']:
                vectorstore.add_texts(
                    [ex['question'] for ex in examples], 
                    metadatas=examples
                )
                print("Added examples to vectorstore.")
            else:
                print("Vectorstore already contains examples.")
        except:
            # If get() fails, try adding examples anyway
            vectorstore.add_texts(
                [ex['question'] for ex in examples], 
                metadatas=examples
            )
            print("Added examples to vectorstore.")

        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vectorstore,
            k=2,
            input_keys=['question'],
            example_keys=['question', 'query']
        )
        
        print("Semantic similarity selector initialized.")
        
    except Exception as e:
        print(f"Error initializing vectorstore: {e}")

def build_dynamic_prefix(user_question: str) -> str:
    """Build dynamic prefix with similar examples."""
    if example_selector:
        try:
            selected = example_selector.select_examples({"question": user_question})
            formatted_examples = "\n".join([
                f"Human: {ex['question']}\nAI:\nSQLQuery: {ex['query']}" 
                for ex in selected
            ])
            return f"""User Question to answer: {user_question}

Refer to the most similar example below to answer the user question.

Examples:
{formatted_examples}

Now begin.
"""
        except Exception as e:
            print(f"Error selecting examples: {e}")
    
    # Fallback if example selector fails
    return f"User Question to answer: {user_question}\n\nNow begin."

# System message
system_msg = """
You are a helpful assistant that uses tools to interact with a MySQL database. 
Generate syntactically correct MySQL queries using the tools and return a clear natural language answer. 
Use LIMIT 5 by default, avoid SELECT *, and never hallucinate. 
Use the tools in order: list_tables -> describe_table -> run_sql_query.
"""

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_msg),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Initialize LLM
llm = None
agent_executor = None

def initialize_agent():
    """Initialize the agent executor."""
    global llm, agent_executor
    
    if not database_connected:
        return False
        
    try:
        # Check for OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
            return False
            
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo', 
            temperature=0.2, 
            max_tokens=512,
            openai_api_key=openai_api_key
        )
        
        agent = create_openai_tools_agent(
            llm=llm, 
            tools=tools, 
            prompt=prompt_template
        )
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True
        )
        
        print("Agent executor initialized successfully.")
        return True
        
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return False

# Initialize agent
agent_initialized = initialize_agent()

# Pydantic models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    status: str = "success"

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_connected": database_connected,
        "agent_initialized": agent_initialized
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if not database_connected:
            return ChatResponse(
                answer="Database connection failed. Please check your database configuration.",
                status="error"
            )
            
        if not agent_initialized:
            return ChatResponse(
                answer="Agent not initialized. Please check your OpenAI API key.",
                status="error"
            )
        
        question = request.question.strip()
        if not question:
            return ChatResponse(
                answer="Please provide a valid question.",
                status="error"
            )
        
        # Build input with dynamic prefix
        input_text = build_dynamic_prefix(question)
        
        # Execute the agent
        result = agent_executor.invoke({
            "input": input_text,
            "chat_history": []
        })
        
        answer = result.get("output", "Sorry, I could not generate a response.")
        
        return ChatResponse(answer=answer, status="success")
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return ChatResponse(
            answer=f"An error occurred: {str(e)}",
            status="error"
        )

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "SQL Chatbot API is running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)