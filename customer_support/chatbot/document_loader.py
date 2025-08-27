import hashlib
import os
import re
import logging
from typing import List, Optional, Dict, Any, Union
import PyPDF2
import docx2txt
import pandas as pd

from langchain_community.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    TextLoader, 
    CSVLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NEPADocumentProcessor:
    """Enhanced document processing with deduplication, Q&A extraction, and NEPA-specific optimizations"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Optimal for Q&A content
            chunk_overlap=150,  # Better overlap for context retention
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\nQ:", "\nA:", "\n", ".", "!", "?", ";", ":", " ", ""]
        )
        
        # NEPA-specific keywords for content categorization
        self.nepa_keywords = {
            'products': ['product', 'catalog', 'inventory', 'e-cig', 'vape', 'cigarette', 'cigar', 'tobacco', 'snack', 'beverage'],
            'account': ['account', 'membership', 'login', 'register', 'password', 'username', 'setup'],
            'ordering': ['order', 'cart', 'checkout', 'payment', 'purchase', 'bulk', 'discount'],
            'shipping': ['shipping', 'delivery', 'track', 'ups', 'freight', 'pallet', 'nationwide'],
            'support': ['support', 'contact', 'help', 'phone', '561-684-1107', 'email', 'support@nepawholesale.com'],
            'returns': ['return', 'refund', 'exchange', 'damaged', 'warranty', 'policy'],
            'business': ['wholesale', 'retail', 'convenience', 'smoke shop', 'vape shop', 'distributor']
        }
    
    def process_documents(self, raw_documents: List[Document]) -> List[Document]:
        """Process documents with NEPA-specific enhancements and deduplication"""
        
        # Step 1: Extract Q&A pairs if present
        qa_docs = self.extract_qa_pairs(raw_documents)
        logger.info(f"Extracted Q&A pairs: {len(qa_docs)} pairs found")
        
        # Step 2: Clean and normalize content
        cleaned_docs = self.clean_documents(raw_documents + qa_docs)
        logger.info(f"Cleaned documents: {len(raw_documents)} -> {len(cleaned_docs)}")
        
        # Step 3: Split into chunks with NEPA-aware splitting
        chunked_docs = []
        for doc in cleaned_docs:
            if doc.metadata.get('content_type') == 'qa_pair':
                # Keep Q&A pairs intact
                chunked_docs.append(doc)
            else:
                chunks = self.smart_split_document(doc)
                chunked_docs.extend(chunks)
        logger.info(f"Created chunks: {len(chunked_docs)}")
        
        # Step 4: Remove duplicates
        unique_docs = self.remove_duplicates(chunked_docs)
        logger.info(f"After deduplication: {len(unique_docs)} unique chunks")
        
        # Step 5: Enhance metadata with NEPA-specific categories
        enhanced_docs = self.enhance_nepa_metadata(unique_docs)
        
        return enhanced_docs
    
    def extract_qa_pairs(self, documents: List[Document]) -> List[Document]:
        """Extract Q&A pairs from NEPA knowledge base content"""
        qa_docs = []
        
        for doc in documents:
            content = doc.page_content
            
            # Pattern to match Q: ... A: ... format
            qa_pattern = r'Q:\s*(.*?)\s*A:\s*(.*?)(?=\nQ:|$)'
            matches = re.findall(qa_pattern, content, re.DOTALL | re.IGNORECASE)
            
            for question, answer in matches:
                question = question.strip()
                answer = answer.strip()
                
                if len(question) > 10 and len(answer) > 10:  # Filter out very short Q&A
                    qa_content = f"Q: {question}\nA: {answer}"
                    
                    qa_doc = Document(
                        page_content=qa_content,
                        metadata={
                            **doc.metadata,
                            'content_type': 'qa_pair',
                            'question': question,
                            'answer': answer,
                            'qa_category': self.categorize_qa_content(qa_content)
                        }
                    )
                    qa_docs.append(qa_doc)
        
        return qa_docs
    
    def categorize_qa_content(self, content: str) -> str:
        """Categorize Q&A content based on NEPA business domains"""
        content_lower = content.lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.nepa_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return the highest scoring category
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'general'
    
    def smart_split_document(self, doc: Document) -> List[Document]:
        """Smart document splitting that preserves NEPA content structure"""
        content = doc.page_content
        
        # If content contains structured sections, split by sections first
        if self.has_structured_sections(content):
            return self.split_by_sections(doc)
        
        # Otherwise use standard text splitting
        return self.text_splitter.split_documents([doc])
    
    def has_structured_sections(self, content: str) -> bool:
        """Check if content has structured sections (headers, categories, etc.)"""
        section_indicators = [
            r'^[A-Z][A-Za-z\s&]+$',  # All caps section headers
            r'^\w+\s+Information$',   # "Product Information", "Account Information"
            r'^\w+\s+\&\s+\w+$',     # "Returns & Refunds"
        ]
        
        lines = content.split('\n')
        section_count = 0
        
        for line in lines:
            line = line.strip()
            if line and any(re.match(pattern, line) for pattern in section_indicators):
                section_count += 1
        
        return section_count >= 2
    
    def split_by_sections(self, doc: Document) -> List[Document]:
        """Split document by logical sections while preserving context"""
        content = doc.page_content
        lines = content.split('\n')
        
        sections = []
        current_section = []
        current_header = ""
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this is a section header
            if self.is_section_header(line_stripped):
                # Save previous section
                if current_section:
                    section_content = '\n'.join(current_section).strip()
                    if section_content:
                        sections.append((current_header, section_content))
                
                # Start new section
                current_header = line_stripped
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add the last section
        if current_section:
            section_content = '\n'.join(current_section).strip()
            if section_content:
                sections.append((current_header, section_content))
        
        # Create documents for each section
        section_docs = []
        for header, section_content in sections:
            if len(section_content) > 50:  # Skip very short sections
                section_doc = Document(
                    page_content=section_content,
                    metadata={
                        **doc.metadata,
                        'section_header': header,
                        'content_type': 'section'
                    }
                )
                
                # Further split if section is too large
                if len(section_content) > 1500:
                    subsections = self.text_splitter.split_documents([section_doc])
                    section_docs.extend(subsections)
                else:
                    section_docs.append(section_doc)
        
        return section_docs if section_docs else self.text_splitter.split_documents([doc])
    
    def is_section_header(self, line: str) -> bool:
        """Identify if a line is a section header"""
        if not line or len(line) < 3:
            return False
        
        # Common patterns for NEPA knowledge base headers
        header_patterns = [
            r'^[A-Z][A-Za-z\s&]+$',           # All caps or title case
            r'^\w+\s+Information$',            # "Product Information"
            r'^\w+\s+\&\s+\w+$',             # "Returns & Refunds"
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # "Customer Support"
        ]
        
        return any(re.match(pattern, line) for pattern in header_patterns)
    
    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """Clean document content with NEPA-specific cleaning"""
        cleaned = []
        
        for doc in documents:
            content = doc.page_content
            
            # Remove excessive whitespace and normalize
            content = re.sub(r'\n{3,}', '\n\n', content)
            content = re.sub(r'\s+', ' ', content)
            
            # Remove repetitive NEPA-specific patterns
            content = self.remove_nepa_repetitive_patterns(content)
            
            # Skip very short chunks (likely noise)
            if len(content.strip()) < 30:
                continue
            
            # Create cleaned document
            cleaned_doc = Document(
                page_content=content.strip(),
                metadata=doc.metadata.copy()
            )
            cleaned.append(cleaned_doc)
        
        return cleaned
    
    def remove_nepa_repetitive_patterns(self, content: str) -> str:
        """Remove repetitive patterns specific to NEPA content"""
        repetitive_patterns = [
            r"How else can I assist you with NepaWholesale\?",
            r"Is there anything specific you'd like to know about.*?",
            r"Let me know if you need any other information",
            r"Contact us at 561-684-1107.*?",
            r"We're here to help.*?",
            r"NEPA Wholesale.*?established.*?2009",  # Avoid repetitive company intro
        ]
        
        for pattern in repetitive_patterns:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def remove_duplicates(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents with enhanced similarity detection"""
        unique_docs = []
        seen_hashes = set()
        content_similarity_threshold = 0.85  # Slightly lower for Q&A content
        
        for doc in documents:
            # Create a hash of the normalized content
            normalized_content = self.normalize_content(doc.page_content)
            content_hash = hashlib.md5(normalized_content.encode()).hexdigest()
            
            # Check for exact duplicates first
            if content_hash not in seen_hashes:
                # Check for near-duplicates
                is_duplicate = False
                for existing_doc in unique_docs[-50:]:  # Only check recent docs for efficiency
                    similarity = self.calculate_similarity(
                        normalized_content, 
                        self.normalize_content(existing_doc.page_content)
                    )
                    if similarity > content_similarity_threshold:
                        # Prefer Q&A pairs over regular content
                        if (doc.metadata.get('content_type') == 'qa_pair' and 
                            existing_doc.metadata.get('content_type') != 'qa_pair'):
                            # Replace existing with Q&A pair
                            unique_docs.remove(existing_doc)
                            break
                        else:
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    seen_hashes.add(content_hash)
                    unique_docs.append(doc)
        
        return unique_docs
    
    def normalize_content(self, content: str) -> str:
        """Normalize content for duplicate detection"""
        # Convert to lowercase and remove extra spaces
        normalized = re.sub(r'\s+', ' ', content.lower())
        
        # Remove punctuation for better comparison
        import string
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))
        
        # Remove common NEPA-specific words that might cause false positives
        common_words = ['nepa', 'wholesale', 'contact', 'phone', 'email', 'support']
        for word in common_words:
            normalized = normalized.replace(word, '')
        
        return normalized.strip()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def enhance_nepa_metadata(self, documents: List[Document]) -> List[Document]:
        """Enhance document metadata with NEPA-specific information"""
        enhanced = []
        
        for doc in documents:
            # Add NEPA-specific content categorization
            if 'qa_category' not in doc.metadata:
                doc.metadata['nepa_category'] = self.categorize_qa_content(doc.page_content)
            
            # Add priority level for customer support
            doc.metadata['priority'] = self.determine_priority(doc.page_content)
            
            # Extract contact information
            contact_info = self.extract_contact_info(doc.page_content)
            if contact_info:
                doc.metadata['has_contact_info'] = True
                doc.metadata.update(contact_info)
            
            # Add keywords for better matching
            keywords = self.extract_nepa_keywords(doc.page_content)
            doc.metadata['keywords'] = keywords
            
            # Add content length and complexity
            doc.metadata['content_length'] = len(doc.page_content)
            doc.metadata['word_count'] = len(doc.page_content.split())
            
            enhanced.append(doc)
        
        return enhanced
    
    def determine_priority(self, content: str) -> str:
        """Determine priority level for customer support scenarios"""
        content_lower = content.lower()
        
        high_priority_terms = [
            'urgent', 'emergency', 'immediately', 'asap', 'problem', 'issue',
            'damaged', 'wrong', 'error', 'complaint', 'locked', 'suspended'
        ]
        
        medium_priority_terms = [
            'help', 'support', 'question', 'how to', 'can you', 'need',
            'order', 'delivery', 'shipping', 'payment'
        ]
        
        if any(term in content_lower for term in high_priority_terms):
            return 'high'
        elif any(term in content_lower for term in medium_priority_terms):
            return 'medium'
        else:
            return 'low'
    
    def extract_contact_info(self, content: str) -> Dict[str, Any]:
        """Extract contact information from content"""
        contact_info = {}
        
        # Phone number pattern
        phone_pattern = r'561-684-1107'
        if re.search(phone_pattern, content):
            contact_info['phone'] = '561-684-1107'
        
        # Email pattern
        email_pattern = r'support@nepawholesale\.com'
        if re.search(email_pattern, content):
            contact_info['email'] = 'support@nepawholesale.com'
        
        return contact_info
    
    def extract_nepa_keywords(self, content: str) -> List[str]:
        """Extract NEPA-specific keywords from content"""
        found_keywords = []
        content_lower = content.lower()
        
        # Flatten all keywords from categories
        all_keywords = []
        for keywords in self.nepa_keywords.values():
            all_keywords.extend(keywords)
        
        # Add some additional specific keywords
        all_keywords.extend([
            'membership', 'catalog', 'pricing', 'bulk discount', 'loyalty',
            'nationwide delivery', 'florida', 'palm beach', 'miami', 'orlando', 'dallas'
        ])
        
        for keyword in all_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        return list(set(found_keywords))  # Remove duplicates


class NEPADocumentLoader:
    """Enhanced document loader specifically designed for NEPA Wholesale knowledge base"""
    
    def __init__(self):
        self.processor = NEPADocumentProcessor()
        self.supported_extensions = {
            '.pdf': (PyPDFLoader, self.validate_pdf),
            '.docx': (Docx2txtLoader, self.validate_docx),
            '.doc': (Docx2txtLoader, self.validate_docx),
            '.csv': (CSVLoader, self.validate_csv),
            '.txt': (TextLoader, self.validate_txt),
            '.md': (UnstructuredMarkdownLoader, self.validate_txt),
            '.markdown': (UnstructuredMarkdownLoader, self.validate_txt),
            '.xlsx': (UnstructuredExcelLoader, self.validate_excel),
            '.xls': (UnstructuredExcelLoader, self.validate_excel)
        }
    
    def validate_pdf(self, file_path: str) -> bool:
        """Validate PDF file"""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if len(reader.pages) == 0:
                    logger.warning(f"PDF {file_path} has no pages.")
                    return False
                
                # Check first few pages for content
                text = ""
                for page in reader.pages[:min(3, len(reader.pages))]:
                    text += page.extract_text() or ""
                
                if not text.strip():
                    logger.warning(f"PDF {file_path} has no extractable text.")
                    return False
                return True
        except Exception as e:
            logger.error(f"PDF validation failed for {file_path}: {str(e)}")
            return False

    def validate_docx(self, file_path: str) -> bool:
        """Validate DOCX file"""
        try:
            text = docx2txt.process(file_path)
            if not text.strip():
                logger.warning(f"DOCX {file_path} has no extractable text.")
                return False
            return True
        except Exception as e:
            logger.error(f"DOCX validation failed for {file_path}: {str(e)}")
            return False

    def validate_csv(self, file_path: str) -> bool:
        """Validate CSV file"""
        try:
            df = pd.read_csv(file_path, nrows=5)
            if df.empty:
                logger.warning(f"CSV {file_path} is empty.")
                return False
            return True
        except Exception as e:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        logger.warning(f"CSV {file_path} is empty.")
                        return False
                    return True
            except Exception as e2:
                logger.error(f"CSV validation failed for {file_path}: {str(e2)}")
                return False

    def validate_txt(self, file_path: str) -> bool:
        """Validate text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    logger.warning(f"Text file {file_path} is empty.")
                    return False
                return True
        except Exception as e:
            logger.error(f"Text file validation failed for {file_path}: {str(e)}")
            return False

    def validate_excel(self, file_path: str) -> bool:
        """Validate Excel file"""
        try:
            df = pd.read_excel(file_path, nrows=5)
            if df.empty:
                logger.warning(f"Excel {file_path} is empty.")
                return False
            return True
        except Exception as e:
            logger.error(f"Excel validation failed for {file_path}: {str(e)}")
            return False

    def load_single_document(self, file_path: str) -> List[Document]:
        """Load a single document file with NEPA-specific processing"""
        if not os.path.isfile(file_path):
            logger.error(f"'{file_path}' is not a valid file.")
            return []
        
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension not in self.supported_extensions:
            logger.error(f"Unsupported file extension: {extension} for {file_path}")
            return []
        
        loader_class, validator = self.supported_extensions[extension]
        logger.info(f"Attempting to load NEPA document: {file_path} with {loader_class.__name__}")
        
        try:
            # Validate file content
            logger.info(f"Validating file: {file_path}")
            if not validator(file_path):
                logger.error(f"Validation failed for {file_path}")
                return []
            
            # Load document with appropriate loader
            logger.info(f"Loading document: {file_path}")
            if extension == '.txt':
                loader = loader_class(file_path, encoding='utf-8')
            else:
                loader = loader_class(file_path)
            
            docs = loader.load()
            logger.info(f"Raw documents loaded: {len(docs)} from {file_path}")
            
            if not docs:
                logger.error(f"No content loaded from {file_path}")
                return []
            
            # Add basic metadata
            for doc in docs:
                doc.metadata.update({
                    'source': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_type': extension.replace('.', ''),
                    'is_nepa_content': True
                })
            
            logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")
            return docs
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}", exc_info=True)
            return []
        
    def load_documents(self, directory: str) -> List[Document]:
        """Load and process NEPA documents from a directory"""
        if not os.path.exists(directory):
            logger.error(f"Directory '{directory}' does not exist.")
            return []

        all_documents = []
        
        try:
            # Walk through directory and load all supported files
            for root, _, files in os.walk(directory):
                if not files:
                    logger.warning(f"No files found in directory: {root}")
                    continue
                
                for file in files:
                    file_path = os.path.join(root, file)
                    docs = self.load_single_document(file_path)
                    all_documents.extend(docs)

            logger.info(f"Total raw NEPA documents loaded: {len(all_documents)}")
            
            # Process documents with NEPA-specific enhancements
            if all_documents:
                processed_documents = self.processor.process_documents(all_documents)
                logger.info(f"Final processed NEPA documents: {len(processed_documents)} chunks")
                
                # Log some statistics
                self.log_processing_stats(processed_documents)
                
                return processed_documents
            
            return []

        except Exception as e:
            logger.error(f"Error walking directory {directory}: {str(e)}")
            return []
    
    def log_processing_stats(self, documents: List[Document]):
        """Log processing statistics for NEPA documents"""
        qa_pairs = sum(1 for doc in documents if doc.metadata.get('content_type') == 'qa_pair')
        categories = {}
        priorities = {}
        
        for doc in documents:
            # Count categories
            category = doc.metadata.get('nepa_category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            # Count priorities
            priority = doc.metadata.get('priority', 'unknown')
            priorities[priority] = priorities.get(priority, 0) + 1
        
        logger.info(f"Processing Statistics:")
        logger.info(f"  - Q&A pairs extracted: {qa_pairs}")
        logger.info(f"  - Categories: {dict(categories)}")
        logger.info(f"  - Priority distribution: {dict(priorities)}")


# Convenience functions for backward compatibility and easy usage
def load_nepa_documents(directory: str) -> List[Document]:
    """Load and process NEPA documents from a directory"""
    loader = NEPADocumentLoader()
    return loader.load_documents(directory)

def load_single_nepa_document(file_path: str) -> List[Document]:
    """Load a single NEPA document file"""
    loader = NEPADocumentLoader()
    return loader.load_single_document(file_path)

def create_nepa_knowledge_base(file_paths: Union[str, List[str]]) -> List[Document]:
    """Create NEPA knowledge base from single file or directory"""
    loader = NEPADocumentLoader()
    
    if isinstance(file_paths, str):
        if os.path.isdir(file_paths):
            return loader.load_documents(file_paths)
        else:
            return loader.load_single_document(file_paths)
    else:
        # Handle list of file paths
        all_docs = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                docs = loader.load_documents(file_path)
            else:
                docs = loader.load_single_document(file_path)
            all_docs.extend(docs)
        
        # Process all documents together for better deduplication
        if all_docs:
            return loader.processor.process_documents(all_docs)
        return []


# Example usage and testing
if __name__ == "__main__":
    # Test the NEPA document loader
    print("NEPA Wholesale Document Loader Test")
    print("=" * 40)
    
    # Test with a directory
    try:
        docs = load_nepa_documents("./nepa_documents")
        print(f"✓ Loaded {len(docs)} NEPA document chunks")
        
        # Print sample document info
        if docs:
            sample_doc = docs[0]
            print(f"\nSample document:")
            print(f"Content preview: {sample_doc.page_content[:200]}...")
            print(f"Metadata: {sample_doc.metadata}")
            
            # Show Q&A pairs if any
            qa_docs = [doc for doc in docs if doc.metadata.get('content_type') == 'qa_pair']
            if qa_docs:
                print(f"\n✓ Found {len(qa_docs)} Q&A pairs")
                print(f"Sample Q&A: {qa_docs[0].page_content[:200]}...")
    
    except Exception as e:
        print(f"✗ Error testing document loader: {str(e)}")
    
    print("\nNEPA Document Loader ready for use!")