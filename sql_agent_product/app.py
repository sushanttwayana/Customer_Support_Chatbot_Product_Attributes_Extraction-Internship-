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
import streamlit as st
from fastapi import FastAPI
import sqlalchemy

st.set_page_config(page_title = "Customer Support Agent",layout="centered")
st.title("CUSTOMER SUPPORT AGENT")

app = FastAPI()

@st.cache_resource
def get_llm_embedding_model():
    ### STEP-2 INITIALIZE THE LLM AND EMBEDDING MODEL
    # llm = ChatGroq(model = "Llama-3.3-70b-Versatile",temperature=0.3,max_tokens = 300)
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',temperature=0.2)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm , embedding_model

@st.cache_resource
def get_vectorstore():
    ##### STEP-3 CREATING FEW-SHOT EXAMPLES
    
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
    }
    ]


    ##### STEP-4 CREATING THE VECTORSTORE
    vectorstore = Chroma(
        embedding_function = embedding_model, # Embedding model
        collection_name = "example_collection", # Table name in vectorstore
        persist_directory="./chroma_db" 
    )
    vectorstore.add_texts([ex['question'] for ex in examples],metadatas=examples) # Addind the examples in vectorstore 
    return vectorstore


### STEP-1 CONNECT TO DATABASE
# Credentials such as username,password,database name etc 
username = 'readonly_user'
password = 'prabal9869'
host = '127.0.0.1'
db_name = 'nepa_wholesale'
mysql_uri = f"mysql+pymysql://{username}:{password}@{host}/{db_name}"
db = SQLDatabase.from_uri(mysql_uri,sample_rows_in_table_info=2)


llm , embedding_model = get_llm_embedding_model()
vectorstore = get_vectorstore()



#### STEP-5 DEFINING THE EXAMPLES SELECTOR FROM VECTORSTORE
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=1 , 
    input_keys = ['question'],   # Which key to use for similarity search
    example_keys=['question','query'] # Which keys to return
)

### STEP-6 BUILD DYNAMIC PROMPT WITH "PROMPT + EXAMPLES"
def build_dynamic_prefix(user_question: str) -> str:
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

### STEP-7 USING CRREATE_SQL_AGENT WITH AGENT_TYPE AS "OPENAI_TOOL"
# tool to get the list of tables from database
@tool
def list_tables(_: str = "") -> str:
    """Pass an empty string to get the list of available tables and it returns a list of usable tables ."""
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
Your response should be like a human customer support agent.
Rules:

1. Use tools in this order: list_tables → describe_table → run_sql_query.
2. Only handle simple greetings like "Hi" or "Hello" without using tools.
3. Strictly For SQL queries, use only SELECT statements. Do not execute queries queries such as INSERT, UPDATE, DELETE,ALTER ,DROP etc that modifies the schema of the database etc.
4. Do not show SQL queries in the final user response.
5. Retrieve a maximum of 5 results using LIMIT unless the user specifies otherwise.
6. Never query all columns; retrieve only the necessary ones.
7. Do not hallucinate answers. If data is not found, respond with: "Sorry I am unable to answer your question."
8. Always handle follow-up questions based on previous chat history and use tools if needed.

Available tools:

1. list_tables – Lists all tables.
2. describe_table – Describes schema of a table.
3. run_sql_query – Executes a raw SQL SELECT query.

Use the tools smartly to identify the right table, understand the schema, generate the right query, and answer concisely.
Below given is the past one conversation between Human and You (AI).
"""
# Trim messages to keep last 2 conversation
def conversational_window_memory(messages):
    print(messages)
    selected_msg = trim_messages(
    messages,
    token_counter=len,  
    max_tokens=2,  
    strategy="last",
    
    start_on="human",

    include_system=True,
    allow_partial=False,
    )
    return selected_msg

tools = [list_tables,describe_table,run_sql_query]

def initialize_agent(prompt):
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor2 = AgentExecutor(agent=agent, tools=tools, verbose=True,return_intermediate_steps=True)
    return agent , agent_executor2

##### STEP - CREATING THE STREAMLIT APP

# Initilaize the message history:
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previously generated messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("User").markdown(msg.content)
    elif isinstance(msg,AIMessage):
        st.chat_message("Assistant").markdown(msg.content)

@app.get("/home")
def get_home():
    return {"message":"From streamlit to fastapi"}


# Take user input
if user_question :=st.chat_input("Welcome to Nepa-Wholesale customer agent"):
    st.session_state.messages.append(user_question)
    st.chat_message('user').markdown(user_question)

    dynamic_prefix = build_dynamic_prefix(user_question)

    prompt = ChatPromptTemplate.from_messages([          # Prompt to agent that conist of : system prompt + chat_history + user_question and dynamic example + Empty agent scratchpad
        ("system", system_msg),

        MessagesPlaceholder(variable_name = "chat_history"),

        ("human",dynamic_prefix),
        MessagesPlaceholder(variable_name="agent_scratchpad") ,
        ("human",user_question) 
    ])
    agent,agent_executor = initialize_agent(prompt) 

    with st.spinner("Generating response..."):
        try:
            response = agent_executor.invoke({               # Invoke the agent with empty "" input and chat_history
                            "chat_history" : st.session_state.chat_history,
                            "input" : user_question
                            })
            
            st.session_state.messages.append(HumanMessage(content=user_question))
            st.session_state.messages.append(AIMessage(content = response['output']))
            st.chat_message('assistant').markdown(response['output'])
            st.session_state.chat_history = conversational_window_memory(st.session_state.messages)

        except sqlalchemy.exc.OperationalError as e:
             
             st.chat_message('assistant').markdown(f"You do not have access to perform this operation .")
             st.session_state.messages.append(HumanMessage(content=user_question))
             st.session_state.messages.append(AIMessage(content ="You do not have access to perform this operation ." ))
             st.session_state.chat_history = conversational_window_memory(st.session_state.messages)