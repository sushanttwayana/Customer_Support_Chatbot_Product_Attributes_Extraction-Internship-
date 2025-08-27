# # chatbot/rag_system.py
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.schema import Document
# import logging
# import re
# from typing import List

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def create_vector_store(documents, embeddings):
#     """Create vector store with better configuration"""
#     logger.info("Creating vector store...")
#     try:
#         # Remove duplicate documents
#         unique_docs = remove_duplicate_documents(documents)
#         logger.info(f"Removed duplicates: {len(documents)} -> {len(unique_docs)} documents")
        
#         vector_store = Chroma.from_documents(
#             documents=unique_docs,
#             embedding=embeddings,
#             collection_name="nepawholesale_docs",
#             persist_directory="./chroma_db"
#         )
#         vector_store.persist()
#         logger.info("Vector store created successfully")
#         return vector_store
#     except Exception as e:
#         logger.error(f"Error creating vector store: {str(e)}")
#         raise

# def remove_duplicate_documents(documents: List[Document]) -> List[Document]:
#     """Remove duplicate documents based on content"""
#     seen_content = set()
#     unique_docs = []
    
#     for doc in documents:
#         # Create a hash of the content to check for duplicates
#         content_hash = hash(doc.page_content.strip())
#         if content_hash not in seen_content:
#             seen_content.add(content_hash)
#             unique_docs.append(doc)
    
#     return unique_docs

# def setup_rag_chain(vector_store, llm):
#     """Setup RAG chain with improved retrieval and prompting"""
#     logger.info("Setting up RAG chain...")
#     try:
#         # Enhanced retriever with better search parameters
#         retriever = vector_store.as_retriever(
#             search_type="mmr",  # Maximum Marginal Relevance for diversity
#             search_kwargs={
#                 "k": 3,  # Reduced from 5 to avoid too much context
#                 "fetch_k": 10,  # Fetch more candidates before MMR filtering
#                 "lambda_mult": 0.7  # Balance between relevance and diversity
#             }
#         )
        
#         # Enhanced prompt template
#         prompt_template = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""You are a professional customer support agent for NepaWholesale. 

# CONTEXT FROM DOCUMENTS:
# {context}

# CUSTOMER QUESTION: {question}

# INSTRUCTIONS:
# 1. Use the provided context to answer accurately and specifically
# 2. If the context contains specific details (prices, policies, procedures), include them
# 3. Be conversational and helpful, not robotic
# 4. If the question is unrelated to NepaWholesale, politely redirect
# 5. Don't repeat "How else can I assist you with NepaWholesale?" - be natural
# 6. Keep responses concise but informative

# RESPONSE:"""
#         )

#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True,
#             chain_type_kwargs={
#                 "prompt": prompt_template,
#                 "verbose": False  # Reduced verbosity
#             },
#             input_key="question"
#         )
        
#         logger.info("RAG chain setup complete")
#         return qa_chain
#     except Exception as e:
#         logger.error(f"Error setting up RAG chain: {str(e)}")
#         raise

# def format_context_for_llm(source_documents: List[Document]) -> str:
#     """Format retrieved documents to avoid repetition and improve quality"""
#     unique_content = []
#     seen_content = set()
    
#     for doc in source_documents:
#         content = doc.page_content.strip()
#         # Simple deduplication
#         if content not in seen_content:
#             seen_content.add(content)
#             unique_content.append(content)
    
#     return "\n\n".join(unique_content)

# class EnhancedRAGChain:
#     """Enhanced RAG chain with better context handling"""
    
#     def __init__(self, vector_store, llm):
#         self.vector_store = vector_store
#         self.llm = llm
#         self.retriever = vector_store.as_retriever(
#             search_type="mmr",
#             search_kwargs={
#                 "k": 3,
#                 "fetch_k": 10,
#                 "lambda_mult": 0.7
#             }
#         )
        
#         self.prompt_template = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""You are a professional customer support agent for NepaWholesale.

# RELEVANT INFORMATION:
# {context}

# CUSTOMER: {question}

# Please provide a helpful, accurate response based on the information above. If the question is about NepaWholesale products, services, or policies, use the specific details from the context. If it's unrelated, politely redirect to NepaWholesale topics.

# Be conversational and natural - avoid repetitive phrases."""
#         )
    
#     def invoke(self, input_dict):
#         """Custom invoke method with better context handling"""
#         try:
#             question = input_dict.get("question", "")
            
#             # Retrieve relevant documents
#             docs = self.retriever.get_relevant_documents(question)
            
#             # Format context with deduplication
#             context = self.format_unique_context(docs)
            
#             # Create prompt
#             prompt = self.prompt_template.format(
#                 context=context,
#                 question=question
#             )
            
#             # Get LLM response
#             response = self.llm.invoke(prompt)
#             result = response.content if hasattr(response, 'content') else str(response)
            
#             return {
#                 "question": question,
#                 "result": result,
#                 "source_documents": docs
#             }
            
#         except Exception as e:
#             logger.error(f"Error in enhanced RAG chain: {str(e)}")
#             return {
#                 "question": input_dict.get("question", ""),
#                 "result": "I apologize, but I'm having trouble processing your request. Please try again or contact our support team.",
#                 "source_documents": []
#             }
    
#     def format_unique_context(self, docs: List[Document]) -> str:
#         """Format context by removing duplicates and organizing content"""
#         if not docs:
#             return "No relevant information found."
        
#         # Group content by type (Q&A, policies, etc.)
#         qa_pairs = []
#         other_content = []
        
#         seen_content = set()
        
#         for doc in docs:
#             content = doc.page_content.strip()
            
#             # Skip if we've seen this exact content
#             if content in seen_content:
#                 continue
#             seen_content.add(content)
            
#             # Check if it's a Q&A format
#             if re.search(r'^Q:', content, re.MULTILINE) and re.search(r'^A:', content, re.MULTILINE):
#                 qa_pairs.append(content)
#             else:
#                 other_content.append(content)
        
#         # Format the context
#         formatted_context = []
        
#         if qa_pairs:
#             formatted_context.append("RELEVANT Q&A:")
#             formatted_context.extend(qa_pairs)
        
#         if other_content:
#             if qa_pairs:
#                 formatted_context.append("\nADDITIONAL INFORMATION:")
#             formatted_context.extend(other_content)
        
#         return "\n\n".join(formatted_context)

from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import logging
import re
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_store(documents, embeddings):
    """Create vector store with better configuration"""
    logger.info("Creating vector store...")
    try:
        # Remove duplicate documents
        unique_docs = remove_duplicate_documents(documents)
        logger.info(f"Removed duplicates: {len(documents)} -> {len(unique_docs)} documents")
        
        vector_store = Chroma.from_documents(
            documents=unique_docs,
            embedding=embeddings,
            collection_name="nepawholesale_docs",
            persist_directory="./chroma_db"
        )
        vector_store.persist()
        logger.info("Vector store created successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def remove_duplicate_documents(documents: List[Document]) -> List[Document]:
    """Remove duplicate documents based on content"""
    seen_content = set()
    unique_docs = []
    
    for doc in documents:
        # Create a hash of the content to check for duplicates
        content_hash = hash(doc.page_content.strip())
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)
    
    return unique_docs

def setup_rag_chain(vector_store, llm):
    """Setup RAG chain with improved retrieval and prompting"""
    logger.info("Setting up RAG chain...")
    try:
        # Enhanced retriever with better search parameters
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,  # Increased to retrieve more documents
                "fetch_k": 15,  # Fetch more candidates
                "lambda_mult": 0.7
            }
        )
        
        # Enhanced prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a professional customer support agent for NepaWholesale, a leading distributor of wholesale products.

RELEVANT INFORMATION:
{context}

CUSTOMER QUESTION: {question}

INSTRUCTIONS:
1. Provide a clear, concise, and accurate answer based on the context.
2. Include specific details (e.g., prices, policies, contact info) if available.
3. Be conversational, friendly, and professional; avoid robotic or repetitive phrases.
4. If the context lacks specific details, offer general guidance and suggest contacting support if needed.
5. For unrelated questions, politely redirect to NepaWholesale topics with a friendly tone.
6. Never include 'Agent:' or 'Customer:' in the response.

RESPONSE:"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_template,
                "verbose": False
            },
            input_key="question"
        )
        
        logger.info("RAG chain setup complete")
        return qa_chain
    except Exception as e:
        logger.error(f"Error setting up RAG chain: {str(e)}")
        raise

def format_context_for_llm(source_documents: List[Document]) -> str:
    """Format retrieved documents to avoid repetition and improve quality"""
    unique_content = []
    seen_content = set()
    
    for doc in source_documents:
        content = doc.page_content.strip()
        if content not in seen_content:
            seen_content.add(content)
            unique_content.append(content)
    
    return "\n\n".join(unique_content)

class EnhancedRAGChain:
    """Enhanced RAG chain with better context handling"""
    
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,  # Increased to retrieve more documents
                "fetch_k": 15,
                "lambda_mult": 0.7
            }
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a professional customer support agent for NepaWholesale, specializing in wholesale products and services.

RELEVANT INFORMATION:
{context}

CUSTOMER QUESTION: {question}

INSTRUCTIONS:
1. Answer clearly and concisely using the provided context.
2. Include specific details (e.g., contact info, policies) when available.
3. Be friendly, professional, and conversational; avoid repetitive phrases.
4. If the context is insufficient, provide general guidance and suggest contacting support at 561-684-1107 or support@nepawholesale.com.
5. For unrelated questions, politely explain that you focus on NepaWholesale and suggest relevant topics.
6. Never include 'Agent:' or 'Customer:' in the response.

RESPONSE:"""
        )
    
    def invoke(self, input_dict):
        """Custom invoke method with better context handling"""
        try:
            question = input_dict.get("question", "")
            
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(docs)} documents for question: {question}")
            
            # Format context with deduplication
            context = self.format_unique_context(docs)
            
            # Create prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            # Clean response
            result = self.clean_response(result)
            
            return {
                "question": question,
                "result": result,
                "source_documents": docs
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced RAG chain: {str(e)}", exc_info=True)
            return {
                "question": input_dict.get("question", ""),
                "result": (
                    "I'm sorry, I'm having trouble finding that information. "
                    "Please contact our support team at 561-684-1107 or support@nepawholesale.com for assistance."
                ),
                "source_documents": []
            }
    
    def format_unique_context(self, docs: List[Document]) -> str:
        """Format context by removing duplicates and organizing content"""
        if not docs:
            return "No relevant information found."
        
        qa_pairs = []
        other_content = []
        seen_content = set()
        
        for doc in docs:
            content = doc.page_content.strip()
            if content in seen_content:
                continue
            seen_content.add(content)
            
            if re.search(r'^Q:', content, re.MULTILINE) and re.search(r'^A:', content, re.MULTILINE):
                qa_pairs.append(content)
            else:
                other_content.append(content)
        
        formatted_context = []
        if qa_pairs:
            formatted_context.append("RELEVANT Q&A:")
            formatted_context.extend(qa_pairs)
        
        if other_content:
            if qa_pairs:
                formatted_context.append("\nADDITIONAL INFORMATION:")
            formatted_context.extend(other_content)
        
        return "\n\n".join(formatted_context)

    def clean_response(self, response: str) -> str:
        """Clean the response to remove unwanted prefixes and ensure quality"""
        response = re.sub(r'^(Agent|Customer): ?', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n{3,}', '\n\n', response)
        if not response.endswith(('.', '!', '?')):
            response += "."
        return response.strip()