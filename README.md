#  AI Solutions

This repository houses a suite of AI-powered solutions designed to enhance the e-commerce operations of NepaWholesale. The projects are built using a variety of technologies, including Python, FastAPI, and LangChain, and leverage Large Language Models (LLMs) to automate and improve key business functions.

## Repository Structure

The repository includes three main projects:

1.  **Customer Service Chatbot**: A Retrieval-Augmented Generation (RAG) system for automated customer support.
2.  **Product Attribute Extractor**: A tool for extracting structured information from product descriptions.
3.  **SQL Database Agent**: An LLM-powered agent for natural language querying of product data.

---

## 1. Customer Service Chatbot

This project is a customer support chatbot built on a Retrieval-Augmented Generation (RAG) architecture using LangChain. It is specifically designed to handle general inquiries and technical support questions for NepaWholesale's customers.

### Architecture & Workflow

The chatbot's architecture is a streamlined pipeline that processes user queries and generates context-aware responses.

1.  **User Query**: A user sends a message through a FastAPI interface.
2.  **Query Processing**: The system analyzes the query and routes it to the appropriate module:
    *   **RAG Chain**: For general inquiries and questions that can be answered by the knowledge base.
    *   **Support Agent**: For specialized queries requiring a tailored response, such as account issues or technical problems.
    *   **Unrelated Queries**: These are politely redirected.
3.  **Response Generation**:
    *   **RAG Chain**: Retrieves relevant document chunks from the Chroma vector store, combines them with the query, and uses an LLM (GPT-3.5 Turbo) to generate a factual response.
    *   **Support Agent**: Uses a specific prompt template and keyword analysis to provide step-by-step guidance for support-related issues.
4.  **Output**: The generated response is returned to the user via the API and stored in a SQLite database for future reference and analysis.

### Key Components

*   `main.py`: The central FastAPI application that serves the API, manages user sessions, and routes queries to the correct processing components.
*   `document_loader.py`: Handles the ingestion and processing of knowledge base documents (e.g., PDFs), splitting them into chunks suitable for the RAG system.
*   `rag_system.py`: Implements the RAG pipeline. It manages the Chroma vector store, handles document retrieval using Maximum Marginal Relevance (MMR) search, and sets up the LLM for response generation.
*   `agent.py`: The specialized `CustomerSupportAgent` that provides robust, tailored responses for support-related queries. It incorporates specific logic for common issues like authentication errors or account problems.


## 2. Product Attribute Extractor
This project focuses on leveraging LLMs or rule-based methods (e.g., regex) to extract specific attributes from unstructured product descriptions. This is crucial for normalizing and enriching product data, making it easier to categorize, search, and analyze.

### Features
1. *Automated Extraction: Automatically identifies and extracts key product attributes (e.g., color, size, material, brand) from text.*

2. Hybrid Approach: Can use either LLM-powered parsing for complex, unstructured text or a more deterministic regex-based approach for predictable patterns.

3. Scalability: Designed to process large datasets of product descriptions efficiently.

## 3. SQL Database Agent
This project showcases a LangChain SQL Agent that allows users to interact with a product database using natural language. Instead of writing SQL queries, users can ask questions like "How many products are in the 'Electronics' category?" or "List the top 5 most expensive products."

### How It Works

* Natural Language Query: A user submits a question in plain English.

* LLM-Powered Translation: The LLM translates the user's query into a valid SQL command. This process involves understanding the user's intent, the database schema, and the necessary SQL syntax.

* Database Interaction: The generated SQL query is executed against the database.

* Response Generation: The query results are presented to the user in a clear, text-based format.

**This agent empowers non-technical users to gain insights from the database without needing to learn SQL.**

### Setup & Usage

**Requirements**: You will need an OpenAI API key for the LLM.

**Installation**:

```bash
git clone https://github.com/your-username/nepawholesale-ai-solutions.git
cd nepawholesale-ai-solutions/customer_service_chatbot
pip install -r requirements.txt
