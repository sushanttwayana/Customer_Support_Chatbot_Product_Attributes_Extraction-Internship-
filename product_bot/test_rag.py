from langchain_community.document_loaders import PyPDFLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pypdf_loader():
    file_path = "docs/Customer Support KB Updated.pdf"
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from {file_path}")
        if docs:
            logger.info(f"Sample content: {docs[0].page_content[:200]}")
    except Exception as e:
        logger.error(f"Error loading PDF with PyPDFLoader: {str(e)}")

if __name__ == "__main__":
    test_pypdf_loader()