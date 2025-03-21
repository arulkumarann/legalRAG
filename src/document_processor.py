from typing import List, Dict
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from src.config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
    
    def load_documents(self, directory: str) -> List[Dict]:
        documents = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            logger.error(f"Directory does not exist: {directory}")
            return documents
        
        logger.info(f"Loading documents from {directory}")
        
        for file_path in dir_path.glob("**/*"):
            if not file_path.is_file():
                continue
                
            try:
                if file_path.suffix.lower() == '.pdf':
                    logger.info(f"Loading PDF: {file_path}")
                    loader = PyPDFLoader(str(file_path))
                    documents.extend(loader.load())
                elif file_path.suffix.lower() == '.txt':
                    logger.info(f"Loading text: {file_path}")
                    loader = TextLoader(str(file_path))
                    documents.extend(loader.load())
                elif file_path.suffix.lower() in ('.xlsx', '.xls'):
                    logger.info(f"Loading Excel: {file_path}")
                    loader = UnstructuredExcelLoader(str(file_path))
                    documents.extend(loader.load())
                elif file_path.suffix.lower() in ('.doc', '.docx'):
                    logger.info(f"Loading Word document: {file_path}")
                    loader = UnstructuredExcelLoader(str(file_path))
                    documents.extend(loader.load())
                else:
                    logger.warning(f"Skipping unsupported file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        if not documents:
            logger.warning("No documents to process!")
            return []
        processed_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(processed_docs)} chunks")
        return processed_docs