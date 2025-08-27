import streamlit as st
from langchain_core.messages import trim_messages
from langchain_core.messages import HumanMessage,AnyMessage,AIMessage,ToolMessage,SystemMessage
import requests

st.set_page_config(page_title = "Customer Support Agent",layout="centered")
st.title("CUSTOMER SUPPORT AGENT")

API_URL = "http://127.0.0.1:8000/chat"

def conversational_window_memory(messages):
    print(messages)
    selected_msg = trim_messages(
    messages,
    token_counter=len,  
    max_tokens=3,  
    strategy="last",
    
    start_on="human",

    include_system=True,
    allow_partial=False,
    )
    return selected_msg

def serialize_chat_history(chat_history):
    serialized = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            serialized.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serialized.append({"type": "ai", "content": msg.content})
        
    return serialized


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
        # Only display if it is NOT an SQL message
        if not msg.content.strip().lower().startswith("generated sql query"):
            st.chat_message("Assistant").markdown(msg.content)

if user_question :=st.chat_input("Welcome to Nepa-Wholesale customer agent"):
    st.chat_message("User").markdown(user_question)
    st.session_state.messages.append(HumanMessage(content = user_question))

    input_data ={
        "question":user_question,
        'chat_history' : serialize_chat_history(st.session_state.chat_history)
    }
    try:
        response = requests.post(API_URL,json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.chat_message("Assistant").markdown(result['message'])

             # Append SQL query as another assistant message
            if result['sql_query']:
                sql_msg = f"Generated SQL Query:\n```sql\n{result['sql_query']}\n```"
                st.session_state.messages.append(AIMessage(content=sql_msg))


            st.session_state.messages.append(AIMessage(content = result['message']))

            st.session_state.chat_history = conversational_window_memory(st.session_state.messages)

        else:
            st.chat_message("Assistant").markdown(f"API ERROR {response.status_code} : {response.text}")

    except requests.exceptions.ConnectionError:
        st.chat_message("Assistant").markdown(f"Could not connect to the FastAPI server")

