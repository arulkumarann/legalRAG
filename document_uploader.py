import os
import sys
import logging
import gc
from typing import List, Dict, Optional
from dataclasses import dataclass

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone

from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('document_uploader.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    batch_size: int = 8
    max_length: int = 512
    dimension: int = 768  # all-mpnet-base-v2 dimension


@dataclass
class TextSplitterConfig:
    """Configuration for text splitter"""
    chunk_size: int = 800
    chunk_overlap: int = 200
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class PineconeConfig:
    """Configuration for Pinecone"""
    api_key: str
    index_name: str
    dimension: int = 768
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


class OptimizedEmbeddings(Embeddings):
    """Optimized embedding class with better memory management"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device} for embeddings")
        
        # Load tokenizer immediately, defer model loading until needed
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = None
    
    def _load_model_if_needed(self):
        """Lazy load model only when needed"""
        if self.model is None:
            logger.info(f"Loading model: {self.config.model_name}")
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on model outputs"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        self._load_model_if_needed()
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i+self.config.batch_size]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
                
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
            
            # Free memory after each batch
            del inputs, outputs, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.vstack(all_embeddings).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a query"""
        return self.embed_documents([text])[0]


class DocumentProcessor:
    """Process documents into chunks"""
    
    def __init__(self, config: TextSplitterConfig):
        self.config = config
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators
        )
    
    def process_text(self, text: str, file_path: str) -> List[Document]:
        """Process a text into document chunks"""
        # Extract basic metadata
        metadata = {
            "source": os.path.basename(file_path),
            "filetype": os.path.splitext(file_path)[1].replace(".", "").lower(),
            "full_path": file_path
        }
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            
            doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def process_directory(self, directory_path: str, file_types: List[str] = None) -> List[Document]:
        """Process all text files in a directory"""
        if file_types is None:
            file_types = ['.txt', '.md', '.csv', '.json']
            
        all_documents = []
        
        # Get all matching files
        text_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in file_types):
                    text_files.append(os.path.join(root, file))
        
        # Process each file
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                documents = self.process_text(text, file_path)
                all_documents.extend(documents)
                logger.info(f"Processed {file_path} into {len(documents)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        logger.info(f"Processed {len(text_files)} files into {len(all_documents)} total chunks")
        return all_documents


class PineconeUploader:
    """Uploads documents to Pinecone"""
    
    def __init__(self, config: PineconeConfig, embeddings: Embeddings):
        self.config = config
        self.embeddings = embeddings
        self.pc = Pinecone(api_key=config.api_key)
    
    def _ensure_index_exists(self) -> bool:
        """Create index if it doesn't exist"""
        try:
            # Check if index exists
            try:
                index = self.pc.Index(self.config.index_name)
                stats = index.describe_index_stats()
                logger.info(f"Index {self.config.index_name} exists with {stats.get('total_vector_count', 0)} vectors")
                return True
            except Exception as e:
                if "not found" in str(e).lower() or "404" in str(e):
                    logger.info(f"Creating new index: {self.config.index_name}")
                    # Create new index
                    self.pc.create_index(
                        name=self.config.index_name,
                        dimension=self.config.dimension,
                        metric=self.config.metric,
                        spec={
                            "serverless": {
                                "cloud": self.config.cloud,
                                "region": self.config.region
                            }
                        }
                    )
                    logger.info(f"Index {self.config.index_name} created successfully")
                    return True
                else:
                    logger.error(f"Error checking index: {str(e)}")
                    return False
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False
    
    def upload_documents(self, documents: List[Document], batch_size: int = 50) -> bool:
        """Upload documents to Pinecone"""
        if not documents:
            logger.warning("No documents to upload")
            return False
        
        # Ensure index exists
        if not self._ensure_index_exists():
            logger.error("Failed to create or confirm index")
            return False
        
        total_docs = len(documents)
        logger.info(f"Uploading {total_docs} documents to Pinecone")
        
        # Process in batches
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch_num = i//batch_size + 1
            logger.info(f"Processing batch {batch_num}/{(total_docs-1)//batch_size+1}: docs {i+1} to {end_idx}")
            
            batch_docs = documents[i:end_idx]
            
            try:
                # Upload documents
                if i == 0:
                    # First batch - create vectorstore
                    LangchainPinecone.from_documents(
                        batch_docs,
                        self.embeddings,
                        index_name=self.config.index_name,
                        # pinecone_api_key=self.config.api_key
                    )
                else:
                    # Get index and add documents
                    vectorstore = LangchainPinecone.from_existing_index(
                        index_name=self.config.index_name,
                        embedding=self.embeddings
                    )
                    vectorstore.add_documents(batch_docs)
                
                logger.info(f"Batch {batch_num} uploaded successfully")
            except Exception as e:
                logger.error(f"Error uploading batch {batch_num}: {str(e)}")
                continue
            
            # Clean memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("Document upload completed")
        return True
    
    def clear_index(self) -> bool:
        """Clear all vectors from the index"""
        try:
            index = self.pc.Index(self.config.index_name)
            index.delete(delete_all=True)
            logger.info(f"Index {self.config.index_name} cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
            return False


class OptimizedDocumentUploader:
    """Main document uploader class with optimized memory management"""
    
    def __init__(
        self, 
        pinecone_api_key: str, 
        index_name: str,
        embedding_config: Optional[EmbeddingConfig] = None,
        text_splitter_config: Optional[TextSplitterConfig] = None,
        pinecone_config: Optional[Dict] = None
    ):
        # Set up configurations
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.text_splitter_config = text_splitter_config or TextSplitterConfig()
        
        # Create Pinecone config
        pc_config = {
            "api_key": pinecone_api_key,
            "index_name": index_name,
            "dimension": self.embedding_config.dimension
        }
        if pinecone_config:
            pc_config.update(pinecone_config)
        self.pinecone_config = PineconeConfig(**pc_config)
        
        # Initialize components
        self.embeddings = OptimizedEmbeddings(self.embedding_config)
        self.processor = DocumentProcessor(self.text_splitter_config)
        self.uploader = PineconeUploader(self.pinecone_config, self.embeddings)
    
    def process_and_upload(
        self, 
        directory_path: str, 
        file_types: List[str] = None,
        batch_size: int = 50,
        clear_existing: bool = False
    ) -> bool:
        """Process and upload documents from a directory"""
        # Clear index if requested
        if clear_existing:
            logger.info("Clearing existing index")
            self.uploader.clear_index()
        
        # Process documents
        logger.info(f"Processing documents from {directory_path}")
        documents = self.processor.process_directory(directory_path, file_types)
        
        if not documents:
            logger.warning("No documents were processed for upload")
            return False
        
        # Upload documents
        logger.info(f"Starting upload of {len(documents)} documents")
        success = self.uploader.upload_documents(documents, batch_size)
        
        if success:
            logger.info("Document upload completed successfully")
        else:
            logger.error("Document upload failed")
        
        return success
    
    def process_text_file(
        self, 
        file_path: str,
        batch_size: int = 50
    ) -> bool:
        """Process and upload a single text file"""
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Process text
            documents = self.processor.process_text(text, file_path)
            logger.info(f"Processed {file_path} into {len(documents)} chunks")
            
            # Upload documents
            if documents:
                logger.info(f"Starting upload of {len(documents)} documents")
                return self.uploader.upload_documents(documents, batch_size)
            else:
                logger.warning(f"No documents were processed from {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = "legal-rag"
    DATA_DIR = r"data"
    
    # Create uploader with default settings
    uploader = OptimizedDocumentUploader(
        pinecone_api_key=API_KEY,
        index_name=INDEX_NAME
    )
    
    # Process and upload documents
    uploader.process_and_upload(
        directory_path=DATA_DIR,
        file_types=['.txt', '.md', '.csv', '.json'],
        batch_size=50,
        clear_existing=False
    )