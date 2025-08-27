from langchain.utilities import SQLDatabase  # Connect to the mysql database
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import HumanMessage,AnyMessage,AIMessage,ToolMessage,SystemMessage
from langchain.tools import tool
from langchain.agents import AgentExecutor , create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough,RunnableSequence
from langchain_core.messages import trim_messages
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field
from typing import Annotated
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import os

app = FastAPI()

username = 'root'
password = 'Sushant@45#'
host = '127.0.0.1'
db_name = 'nepa_wholesale'
mysql_uri = f"mysql+pymysql://{username}:{password}@{host}/{db_name}"

examples = [
    {
        'question': "How many products of fronto are there in x?",
        'query': "SELECT COUNT(*) FROM x WHERE Brand LIKE '%FRONTO%';"
    },
    {
        'question': "How many products are there in x with flavor mint",
        'query': "SELECT COUNT(Product_ID) FROM x WHERE Flavor LIKE '%mint%';"
    },
    {
        'question': "How many products are there in x having nicotine less than 5%. List them?",
        'query': "SELECT Product_ID, Display_Name, Nicotine_strength FROM x WHERE CAST(REPLACE(Nicotine_strength, '%', '') AS DECIMAL(3,2)) < 5 LIMIT 5;"
    },
    {
        'question': "Which flavors of x have 'Raspberry' in them?",
        'query': "SELECT Flavor FROM x WHERE Flavor LIKE '%Raspberry%' LIMIT 5;"
    },
    {
        'question': "List x products with nicotine strength of 2%",
        'query': "SELECT Display_Name, Nicotine_strength FROM x WHERE Nicotine_strength LIKE '%2%%' LIMIT 5;"
    },
    {
        'question': "I’m looking for x products from AL CAPONE. What do you have?",
        'query': "SELECT Display_Name, Flavor FROM x WHERE Brand LIKE '%AL CAPONE%' LIMIT 5;"
    },
    {
        'question': "How many unique flavors are offered in x products with 5000 puffs?",
        'query': "SELECT COUNT(DISTINCT Flavor) FROM x WHERE Puff_count = 5000;"
    },
    {
        'question': "I’m looking for x that come in a 12PK. What options do you have?",
        'query': "SELECT Display_Name, Brand, Flavor FROM x WHERE Packet_count LIKE '%12PK%' LIMIT 5;"
    },
    {
        'question': "Get the top 5 x products with the highest puff count.",
        'query': "SELECT Display_Name, Puff_count FROM x ORDER BY Puff_count DESC LIMIT 5;"
    },
    {
        'question': "Which brands offer x products with puff count equal to 5000?",
        'query': "SELECT DISTINCT Brand FROM x WHERE Puff_count = 5000 LIMIT 5;"
    },
    {
        'question': "How many products are there in DEATH ROW DISPOSABLE sub-category with 2% nicotine and 5 pack count in x?",
        'query': "SELECT COUNT(*) FROM x WHERE Product_Sub_Category LIKE '%DEATH ROW DISPOSABLE%' AND Nicotine_strength LIKE '%2%%' AND Pack_count = 5;"
    },
    {
        'question': "How many different x flavors do you offer?",
        'query': "SELECT COUNT(DISTINCT Flavor) FROM x;"
    },
    {
        'question': "Hello, how are you?",
        'query': "Hello, I am fine. How can I help you?"
    },
    {
        'question' : "Show me some more products ?",
        'query' : 'SELECT Display_Name from x LIMIT 5 OFFSET 5; '
    },
    {
        "question": "Get the top 3 brands with the most products in x.",
        "query": "SELECT Brand, COUNT(*) AS product_count FROM x GROUP BY Brand ORDER BY product_count DESC LIMIT 3;"
    }
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


# llm = ChatGroq(model = "Llama-3.3-70b-Versatile",temperature=0.3,max_tokens = 300)
def initialize_db_llm_embedding():
    db = SQLDatabase.from_uri(mysql_uri,sample_rows_in_table_info=2)
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',temperature=0.2, max_tokens = 2100)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return db,llm ,embedding_model

def initialize_vectorstore(embedding_model):

    persist_dir = "chroma_index"
    if os.path.exists(persist_dir):
        print("--- Loading existing VectorStore ---------")
        vectorstore = Chroma(persist_directory=persist_dir , embedding_function=embedding_model)
    else:

        print("---Creating vectorstore for fisrt time ---------")
        vectorstore = Chroma(
            embedding_function = embedding_model, # Embedding model
            collection_name = "example_collection", # Table name in vectorstore
            persist_directory=persist_dir 
        )
        vectorstore.add_texts([ex['question'] for ex in examples],metadatas=examples)

    return vectorstore

def initialize_exampleselector(embedding_model):
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings = embedding_model,
        k=1 , 
        input_keys = ['question'],   # Which key to use for similarity search
        example_keys=['question','query'], # Which keys to return
        vectorstore_cls=Chroma,
    )
    return example_selector

def build_dynamic_prefix(user_question: str,embedding_model) -> str:
    example_selector = initialize_exampleselector(embedding_model)
    selected = example_selector.select_examples({"question": user_question})
    formatted_examples = "\n".join([
        f"Human: {ex['question']}\nAI:\nSQLQuery: {ex['query']}" for ex in selected
    ])
    return f"""Refer to the below given most similar examples to answer the above user question . 

        Examples:
        {formatted_examples}
        ...
        User Question to answer : 
        """
# dynamic_prefix = build_dynamic_prefix("what is my income of last year")


# tool to get the list of tables from database
@tool
def list_tables(_: str = "") -> str:
    """Use this tool to list all table names that exist in the connected MySQL database. Pass an empty string to get the list of available tables
    This is helpful when you need to know what tables are available before writing a SQL query."""
    return str(db.get_table_names())


# tool to get schema of table to be used
@tool
def describe_table(table_name: str) -> str:
    """Use this tool to get the schema (column names and types) of a specific table.Input should be the name of the table as a string. 
    This is useful to understand what data is stored in the table before writing queries."""
    return str(db.get_table_info([table_name]))

# tool to execute the sql query in database
@tool
def run_sql_query(query: str) -> str:
    """Use this tool to run a raw SQL query on the database and return the result.
    Input should be a complete and valid SQL SELECT query as a string. 
    Use this when you already know which table and columns to query."""
    return str(db.run(query))

system_msg = """
You are a helpful assistant that interacts with a MySQL database using tools. Your task is to create syntactically correct MySQL SELECT queries, execute them using tools, and return a well-formatted, natural language response.
Rules:

1. Use tools in this order and do not skip to use any of these tools : list_tables ,describe_table ,run_sql_query.
2. Only handle simple greetings like "Hi" or "Hello" without using tools.
3. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
4. Do not show SQL queries in the final user response.
5. Retrieve a maximum of 5 results using LIMIT clause, unless the user specifies otherwise.
6. Never query all columns; retrieve only the necessary ones.
7. Do not hallucinate answers. If data is not found, respond with: "Sorry I am unable to answer your question."
8. Always handle follow-up questions based on previous chat history and use tools to check the table name , table description .
9. Use like to search for brand instead of '='

Available tools:

1. list_tables – Lists all tables.
2. describe_table – Describes schema of a table.
3. run_sql_query – Executes a raw SQL SELECT query.

Use the tools smartly to identify the right table, understand the schema, generate the right query, and answer concisely as You are a Customer supporter Agent.
Below given is the past one conversation between Human and You (AI).
"""

tools = [list_tables,describe_table,run_sql_query]

db , llm , embedding_model = initialize_db_llm_embedding()
vectorstore = initialize_vectorstore(embedding_model)

def initialize_agent(prompt):
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor2 = AgentExecutor(agent=agent, tools=tools, verbose=True,return_intermediate_steps=True)
    return agent_executor2

# class ChatMessage(BaseModel):
#     # type: str  # either 'human' or 'ai'
#     # content: str
#     HumanMessage()

class pydantic_validator(BaseModel):
    question : Annotated[str ,Field(...,description = "Question from the user")]
    chat_history : list[dict]


def deserialize_chat_history(chat_history):
    messages = []
    for msg in chat_history:
        if msg['type'] == 'human':
            messages.append(HumanMessage(content=msg['content']))
        elif msg['type'] == 'ai':
            messages.append(AIMessage(content=msg['content']))
    return messages

@app.get("/")
def root():
    return {"message":"Root page"}

@app.post("/chat")
def chat_openai(request:pydantic_validator):
    question = request.question.strip()
    chat_history = deserialize_chat_history(request.chat_history)
    dynamic_prefix = build_dynamic_prefix(question,embedding_model)

    prompt = ChatPromptTemplate.from_messages([          # Prompt to agent that conist of : system prompt + chat_history + user_question and dynamic example + Empty agent scratchpad
        ("system", system_msg),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human",dynamic_prefix),
        ("human",question), 
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent_executor = initialize_agent(prompt) 
    response = agent_executor.invoke({    
                            "chat_history" : chat_history,# Invoke the agent with empty "" input and chat_history
                            "input" : question,
                            })
    sql_query = ""


    for step in response.get("intermediate_steps", []):
        action, _ = step
        if hasattr(action, "tool") and action.tool == "run_sql_query":
            tool_input = action.tool_input
            if isinstance(tool_input, dict):
                sql_query = tool_input.get("query", "")
            elif isinstance(tool_input, str):
                sql_query = tool_input
            else:
                sql_query = str(tool_input)
            tool_call_id = getattr(action, "tool_call_id", "")
            break


    
    result = response['output'] 
    return JSONResponse(status_code=200,content={'message':result,'sql_query':sql_query or ""})

