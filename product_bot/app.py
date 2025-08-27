# from langchain.utilities import SQLDatabase  # Connect to the mysql database
from langchain_community.utilities import SQLDatabase
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

st.set_page_config(page_title = "Customer Support Agent",layout="centered")
st.title("CUSTOMER SUPPORT AGENT")

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
        'question': "How many products of fronto are there in tobaccos ?",
        'query': "SELECT COUNT(*) FROM tobaccos_category WHERE Brand LIKE '%FRONTO%';"
    },
    {
        'question': "How many products are there in tobaccos with flavor mint",
        'query': "SELECT COUNT(Product_ID) FROM tobaccos_category WHERE Flavor LIKE '%mint%';"
    },
    {
        'question': "How many products are there in disposable having nicotine less than 5%. list them ?",
        'query': "SELECT Product_ID, Display_Name, Nicotine_strength FROM disposable_category WHERE CAST(REPLACE(Nicotine_strength, '%', '') AS DECIMAL(3,2)) < 5 LIMIT 5;"
    },
    {
        'question': "Which flavors of cigars have 'Raspberry' in them?",
        'query': "SELECT Flavor FROM cigars_category WHERE Flavor LIKE '%Raspberry%' LIMIT 5;"
    },
    {
        'question': "List disposable products with nicotine strength of 2%",
        'query': "SELECT Display_Name, Nicotine_strength FROM disposable_category WHERE Nicotine_strength LIKE '%2%%' LIMIT 5;"
    },
    {
        'question': "I’m looking for cigar products from AL CAPONE. What do you have?",
        'query': "SELECT Display_Name, Flavor FROM cigars_category WHERE Brand LIKE '%AL CAPONE%' LIMIT 5;"
    },
    {
        'question': "How many unique flavors are offered in disposable products with 5000 puffs?",
        'query': "SELECT COUNT(DISTINCT Flavor) FROM disposable_category WHERE Puff_count = 5000;"
    },
    {
        'question': "I’m looking for cigars that come in a 12PK. What options do you have?",
        'query': "SELECT Display_Name, Brand, Flavor FROM cigars_category WHERE Packet_count LIKE '%12PK%' LIMIT 5;"
    },
    {
        'question': "Get the top 5 disposable products with the highest puff count.",
        'query': "SELECT Display_Name, Puff_count FROM disposable_category ORDER BY Puff_count DESC LIMIT 5;"
    },
    {
        'question': "Which brands offer disposable products with puff count equal to 5000?",
        'query': "SELECT DISTINCT Brand FROM disposable_category WHERE Puff_count = 5000 LIMIT 5;"
    },
    {
        'question': "How many products are there in DEATH ROW DISPOSABLE sub-category with 2% nicotine and 5 pack count?",
        'query': "SELECT COUNT(*) FROM disposable_category WHERE Product_Sub_Category LIKE '%DEATH ROW DISPOSABLE%' AND Nicotine_strength LIKE '%2%%' AND Pack_count = 5;"
    },
    {
        'question': "How many different cigar flavors do you offer?",
        'query': "SELECT COUNT(DISTINCT Flavor) FROM cigars_category;"
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
from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase

# Credentials
username = 'root'
password = 'Sushant@45#'  # Original password with special characters
host = '127.0.0.1'
db_name = 'nepa_wholesale'

# URL encode the password to handle special characters
encoded_password = quote_plus(password)

# Create the MySQL URI with encoded password
mysql_uri = f"mysql+pymysql://{username}:{encoded_password}@{host}/{db_name}"

print(f"Encoded URI: {mysql_uri}")

# Create database connection
try:
    db = SQLDatabase.from_uri(mysql_uri, sample_rows_in_table_info=2)
    print("Database connection successful!")
    
    # Test the connection
    tables = db.get_usable_table_names()
    print(f"Available tables: {tables}")
    
except Exception as e:
    print(f"Connection failed: {e}")


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
    return f"""User Question to answer :  {user_question}

Refer to the below given most similar examples to answer the above user question . 

Examples:
{formatted_examples}
...

Now begin.
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

system_msg = """You are a helpful assistant that uses tools to interact with a MySQL database. Create a syntactically correct MySQL query to run, execute the Query and return the final response in natural language back, after proper formatting.
You can skip using tools to handle simple Hi ,Hello and other simple user questions .
You have access to the following tools, Use the 'list_tables' tools to identify the list of available tables and decide which table to use and then use 'describe_table' tool and based on the results from both generate SQL query and use 'run_sql_query' tool.
1. list_tables: List all available tables. 
2. describe_table: Get the schema of a table. 
3. run_sql_query: Run a raw SQL query. 

Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Do not hallicunate and incase you do not find the answer respond with "Sorry I am unable to answer your question".
Strictly do not include the SQL Query in the final response . 
Below given is the chat history betwen the human and the agent. Also handle the follow up question and use the tools provided if you need ."""

# system_msg = """
# You are an intelligent assistant that interacts with a MySQL database using tools. Your goal is to understand the user's question, use the appropriate tools to generate and execute a valid MySQL query, and return a well-formatted natural language answer.

# Available tools:
# 1. list_tables – Get all table names.
# 2. describe_table – Get the schema of a specific table.
# 3. run_sql_query – Execute a raw SQL query.

# Process:
# - First, use list_tables and describe_table to understand the data.
# - Then, generate a syntactically correct MySQL query using only relevant columns.
# - Use LIMIT 5 unless the user explicitly requests a different number of results.
# - Use ORDER BY to surface the most useful or informative data.
# - Never SELECT * or include unnecessary columns.
# - Do not hallucinate. If you cannot answer based on available data, reply: "Sorry, I am unable to answer your question."
# - Handle follow-up questions using previous chat history and tool outputs.

# The following is the ongoing conversation between the user and the assistant.
# """


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

    agent_executor2 = AgentExecutor(agent=agent, tools=tools, verbose=True)
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

# Take user input
if user_question :=st.chat_input("Welcome to Nepa-Wholesale customer agent"):
    st.session_state.messages.append(user_question)
    st.chat_message('user').markdown(user_question)

    dynamic_prefix = build_dynamic_prefix(user_question)

    prompt = ChatPromptTemplate.from_messages([          # Prompt to agent that conist of : system prompt + chat_history + user_question and dynamic example + Empty agent scratchpad
        ("system", system_msg),

        MessagesPlaceholder(variable_name = "chat_history"),

        ("human",dynamic_prefix),
        MessagesPlaceholder(variable_name="agent_scratchpad")  
    ])

    agent,agent_executor = initialize_agent(prompt) 

    with st.spinner("Generating response..."):
     
        response = agent_executor.invoke({               # Invoke the agent with empty "" input and chat_history
                        "chat_history" : st.session_state.chat_history
                        })


    print(prompt)

    st.session_state.messages.append(HumanMessage(content=user_question))
    st.session_state.messages.append(AIMessage(content = response['output']))
    st.chat_message('assistant').markdown(response['output'])
    st.session_state.chat_history = conversational_window_memory(st.session_state.messages)
    




