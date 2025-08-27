# # chatbot/agent.py - Enhanced version
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from pydantic import BaseModel, Field
# from typing import Optional
# import logging

# logger = logging.getLogger(__name__)

# class CustomerSupportAgent(BaseModel):
#     llm: ChatOpenAI = Field(description="The language model for processing queries")
#     prompt_template: PromptTemplate = Field(description="The prompt template for support queries")
#     vector_store: Optional[Chroma] = Field(default=None, description="Vector store for RAG fallback")

#     class Config:
#         arbitrary_types_allowed = True

#     def __init__(self, llm: ChatOpenAI, vector_store: Optional[Chroma] = None):
#         # Enhanced prompt template for better responses
#         prompt_template = PromptTemplate(
#             input_variables=["query", "context"],
#             template="""You are a professional customer support representative for NepaWholesale.

# CONTEXT (if available):
# {context}

# CUSTOMER INQUIRY:
# {query}

# GUIDELINES:
# - Be helpful, professional, and conversational
# - Use specific information from context when available
# - For technical issues, provide step-by-step solutions
# - If you don't have specific information, direct them to appropriate contact methods
# - Keep responses natural and avoid repetitive phrases
# - Focus on resolving the customer's concern

# RESPONSE:"""
#         )
#         super().__init__(llm=llm, prompt_template=prompt_template, vector_store=vector_store)

#     def handle_support_query(self, query: str) -> str:
#         """Handle support queries with context from vector store"""
#         try:
#             context = ""
            
#             # Try to get relevant context from vector store
#             if self.vector_store:
#                 try:
#                     retriever = self.vector_store.as_retriever(
#                         search_type="similarity",
#                         search_kwargs={"k": 2}
#                     )
#                     docs = retriever.get_relevant_documents(query)
#                     if docs:
#                         context = "\n\n".join([doc.page_content for doc in docs[:2]])
#                 except Exception as e:
#                     logger.warning(f"Could not retrieve context: {str(e)}")
            
#             # Format the prompt
#             formatted_prompt = self.prompt_template.format(
#                 query=query,
#                 context=context if context else "No specific documentation available."
#             )
            
#             # Get response from LLM
#             response = self.llm.invoke(formatted_prompt)
#             result = response.content if hasattr(response, 'content') else str(response)
            
#             # Post-process the response
#             return self.clean_response(result)
            
#         except Exception as e:
#             logger.error(f"Error in support query handling: {str(e)}")
#             return "I apologize for the technical difficulty. Please contact our support team directly at 561-684-1107 or support@nepawholesale.com, and they'll be happy to assist you immediately."

#     def clean_response(self, response: str) -> str:
#         """Clean and improve the response quality"""
#         # Remove excessive repetition
#         lines = response.split('\n')
#         cleaned_lines = []
        
#         for line in lines:
#             line = line.strip()
#             if line and line not in cleaned_lines[-3:]:  # Avoid recent repetition
#                 cleaned_lines.append(line)
        
#         cleaned_response = '\n'.join(cleaned_lines)
        
#         # Ensure it doesn't end abruptly
#         if cleaned_response and not cleaned_response.endswith(('.', '!', '?')):
#             cleaned_response += "."
        
#         return cleaned_response

#     def categorize_query(self, query: str) -> str:
#         """Categorize the type of support query"""
#         query_lower = query.lower()
        
#         if any(word in query_lower for word in ['order', 'purchase', 'buy', 'payment']):
#             return 'order'
#         elif any(word in query_lower for word in ['account', 'login', 'register', 'password']):
#             return 'account'
#         elif any(word in query_lower for word in ['website', 'technical', 'error', 'bug', 'loading']):
#             return 'technical'
#         elif any(word in query_lower for word in ['return', 'refund', 'exchange', 'policy']):
#             return 'returns'
#         elif any(word in query_lower for word in ['product', 'inventory', 'stock', 'catalog']):
#             return 'product'
#         else:
#             return 'general'

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)

class CustomerSupportAgent(BaseModel):
    llm: ChatOpenAI = Field(description="The language model for processing queries")
    prompt_template: PromptTemplate = Field(description="The prompt template for support queries")
    vector_store: Optional[Chroma] = Field(default=None, description="Vector store for RAG fallback")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: ChatOpenAI, vector_store: Optional[Chroma] = None):
        prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""You are a professional customer support representative for NepaWholesale, a leading wholesale distributor.

CONTEXT (if available):
{context}

CUSTOMER INQUIRY:
{query}

GUIDELINES:
- Provide a clear, concise, and friendly response using the context when available.
- For technical issues (e.g., account, website), offer step-by-step guidance or suggest contacting support.
- For account issues, include specific steps like contacting support or resetting passwords.
- If information is missing, provide general advice and suggest contacting 561-684-1107 or support@nepawholesale.com.
- Be conversational and professional; avoid repetitive phrases.
- Never include 'Agent:' or 'Customer:' in the response.

RESPONSE:"""
        )
        super().__init__(llm=llm, prompt_template=prompt_template, vector_store=vector_store)

    def handle_support_query(self, query: str) -> str:
        """Handle support queries with context from vector store"""
        try:
            context = ""
            query_category = self.categorize_query(query)
            
            # Retrieve context from vector store
            if self.vector_store:
                try:
                    retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}  # Increased to retrieve more documents
                    )
                    docs = retriever.get_relevant_documents(query)
                    logger.info(f"Retrieved {len(docs)} documents for support query: {query}")
                    if docs:
                        context = "\n\n".join([doc.page_content for doc in docs[:3]])
                except Exception as e:
                    logger.warning(f"Could not retrieve context: {str(e)}")
            
            # Provide specific guidance for account-related issues
            if query_category == 'account':
                context += (
                    "\n\nFor account issues, customers can contact support at 561-684-1107 or "
                    "support@nepawholesale.com to reset passwords or unlock accounts. "
                    "To prevent login issues, ensure correct credentials and avoid multiple failed attempts."
                )
            
            # Format the prompt
            formatted_prompt = self.prompt_template.format(
                query=query,
                context=context if context else "No specific documentation available."
            )
            
            # Get response from LLM
            response = self.llm.invoke(formatted_prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            # Post-process the response
            return self.clean_response(result, query_category)
            
        except Exception as e:
            logger.error(f"Error in support query handling: {str(e)}", exc_info=True)
            if 'account' in query.lower() or 'login' in query.lower():
                return (
                    "I'm sorry, I couldn't find specific details about your account issue. "
                    "To resolve a locked account or login problem, please contact our support team at "
                    "561-684-1107 or support@nepawholesale.com. They can assist with password resets or account recovery. "
                    "To prevent issues, ensure your credentials are correct and avoid multiple failed login attempts."
                )
            return (
                "I apologize for the technical difficulty. Please contact our support team at "
                "561-684-1107 or support@nepawholesale.com for immediate assistance."
            )

    def clean_response(self, response: str, query_category: str) -> str:
        """Clean and improve the response quality"""
        # Remove unwanted prefixes
        response = re.sub(r'^(Agent|Customer): ?', '', response, flags=re.MULTILINE)
        
        # Remove excessive newlines
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Ensure proper ending
        if not response.endswith(('.', '!', '?')):
            response += "."
        
        # Add category-specific guidance if response is too short
        if len(response) < 50:
            if query_category == 'account':
                response += (
                    " For account-related issues, please contact our support team at "
                    "561-684-1107 or support@nepawholesale.com."
                )
            else:
                response += (
                    " For more details, please reach out to our support team at "
                    "561-684-1107 or support@nepawholesale.com."
                )
        
        return response.strip()

    def categorize_query(self, query: str) -> str:
        """Categorize the type of support query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['order', 'purchase', 'buy', 'payment']):
            return 'order'
        elif any(word in query_lower for word in ['account', 'login', 'register', 'password', 'locked']):
            return 'account'
        elif any(word in query_lower for word in ['website', 'technical', 'error', 'bug', 'loading', 'app']):
            return 'technical'
        elif any(word in query_lower for word in ['return', 'refund', 'exchange', 'policy']):
            return 'returns'
        elif any(word in query_lower for word in ['product', 'inventory', 'stock', 'catalog']):
            return 'product'
        else:
            return 'general'