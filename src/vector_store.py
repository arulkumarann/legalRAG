import os
import gc
import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from src.config import config

# Standard HTTP client, not gRPC
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global model and tokenizer instances
_global_model = None
_global_tokenizer = None

def get_global_model_and_tokenizer():
    """Get or initialize the global model and tokenizer."""
    global _global_model, _global_tokenizer
    
    if _global_tokenizer is None:
        logger.info("Initializing global tokenizer")
        _global_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    
    if _global_model is None:
        logger.info("Initializing global model")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device} for embeddings")
        _global_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        _global_model.to(device)
        _global_model.eval()
    
    return _global_model, _global_tokenizer


class GlobalModelHuggingFaceEmbeddings(Embeddings):
    """LangChain compatible embeddings class using global model instance"""
    
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.dimension = 768  # all-mpnet-base-v2 embedding dimension
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Creating embeddings instance using global model (device: {self.device})")
        
        # Get global model and tokenizer
        self.model, self.tokenizer = get_global_model_and_tokenizer()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts):
        """Create embeddings for documents using batching"""
        all_embeddings = []
        
        # Process in batches to reduce memory usage
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
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
        
        return np.vstack(all_embeddings).tolist()  # LangChain expects list of lists
    
    def embed_query(self, text):
        """Embed a single query using document batching method"""
        return self.embed_documents([text])[0]


class VectorStore:
    def __init__(self, batch_size=4):
        logger.info("Initializing VectorStore...")
        # Use globally shared embeddings model
        self.embeddings = GlobalModelHuggingFaceEmbeddings(batch_size=batch_size)
        
        # Initialize the Pinecone client with standard HTTP client
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.INDEX_NAME
        logger.info(f"VectorStore initialized with index name: {self.index_name}")
    
    def _index_exists(self):
        """Helper method to check if index exists, handling different response formats"""
        try:
            indexes = self.pc.list_indexes()
            
            # Debug output to see the actual structure
            logger.info(f"Index list response type: {type(indexes)}")
            
            # The response is now a dictionary with 'indexes' key containing a list of index objects
            if hasattr(indexes, 'indexes') and isinstance(indexes.indexes, list):
                logger.info(f"First index type: {type(indexes.indexes[0]) if indexes.indexes else 'No indexes'}")
                logger.info(f"Indexes: {indexes}")
                
                # Check each index in the list
                for index_info in indexes.indexes:
                    if hasattr(index_info, 'name') and index_info.name == self.index_name:
                        logger.info(f"Found index: {index_info.name}")
                        return True
                        
                logger.info(f"Index '{self.index_name}' not found")
                return False
            
            # Handle string list format (older SDK versions)
            elif isinstance(indexes, list):
                for index_name in indexes:
                    if index_name == self.index_name:
                        logger.info(f"Found index (string format): {index_name}")
                        return True
                logger.info(f"Index '{self.index_name}' not found in list format")
                return False
            
            # Handle direct index object (some SDK versions)
            elif hasattr(indexes, 'name') and indexes.name == self.index_name:
                logger.info(f"Found index (object format): {indexes.name}")
                return True
            
            else:
                logger.info(f"Unrecognized response format or index not found: {indexes}")
                
                # Direct check as a fallback
                try:
                    index = self.pc.Index(self.index_name)
                    stats = index.describe_index_stats()
                    logger.info(f"Index found via direct access: {self.index_name}")
                    return True
                except Exception as direct_error:
                    logger.warning(f"Direct index access failed: {str(direct_error)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error checking if index exists: {str(e)}", exc_info=True)
            
            # Try direct access as a fallback
            try:
                index = self.pc.Index(self.index_name)
                stats = index.describe_index_stats()
                logger.info(f"Index found via direct access: {self.index_name}")
                return True
            except Exception as direct_error:
                logger.warning(f"Direct index access failed: {str(direct_error)}")
                return False
    
    def create_index(self):
        """Create a Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            if not self._index_exists():
                logger.info(f"Creating new Pinecone serverless index: {self.index_name}")
                # Use dictionary format for spec instead of ServerlessSpec class
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # Dimension matching your embedding model
                    metric='cosine',
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
                
            # Double-check that the index exists after creation
            if not self._index_exists():
                logger.warning("Index creation may still be in progress. Wait a few minutes.")
                
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}", exc_info=True)
            raise
    
    def get_index(self):
        """Get the Pinecone index object directly"""
        try:
            # Direct index access - for the latest SDK
            index = self.pc.Index(self.index_name)
            logger.info(f"Retrieved index directly: {self.index_name}")
            return index
        except Exception as e:
            logger.error(f"Error getting index: {str(e)}", exc_info=True)
            raise
    
    def upload_documents(self, documents, batch_size=50):
        """Upload documents to Pinecone index"""
        if not documents:
            logger.warning("No documents to upload to Pinecone!")
            return None
        
        logger.info(f"Uploading {len(documents)} documents to Pinecone in batches...")
        vectorstore = None
        
        # Make sure the index exists
        if not self._index_exists():
            self.create_index()
        
        # Process documents in smaller batches to manage memory usage
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            logger.info(f"Processing batch {i//batch_size + 1}: documents {i+1} to {end_idx}")
            
            batch_docs = documents[i:end_idx]
            
            try:
                # For the first batch, create the vector store; then add documents subsequently
                if i == 0:
                    # The key fix: add pinecone_api_key explicitly
                    vectorstore = LangchainPinecone.from_documents(
                        batch_docs,
                        self.embeddings,
                        index_name=self.index_name,
                        pinecone_api_key=config.PINECONE_API_KEY  # This is crucial!
                    )
                else:
                    vectorstore.add_documents(batch_docs)
                logger.info(f"Batch {i//batch_size + 1} processed successfully")
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}", exc_info=True)
                continue
                
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return vectorstore
    
    def load_index(self):
        """Load existing Pinecone index with compatibility fix for newer Pinecone SDK"""
        try:
            # First, check if index exists
            direct_check = True
            try:
                index = self.pc.Index(self.index_name)
                stats = index.describe_index_stats()
                logger.info(f"Index {self.index_name} exists via direct check")
            except Exception as e:
                logger.warning(f"Direct index check failed: {str(e)}")
                direct_check = False
                
            if direct_check or self._index_exists():
                logger.info(f"Loading existing Pinecone index: {self.index_name}")
                
                # COMPATIBILITY FIX FOR LANGCHAIN
                try:
                    # Try importing the special compatibility class if available
                    try:
                        from langchain_community.vectorstores.pinecone import PineconeCompatibleIndex
                        pc_index = self.pc.Index(self.index_name)
                        pinecone_compatible = PineconeCompatibleIndex(pc_index)
                        
                        # Use direct constructor with compatible index
                        return LangchainPinecone(
                            index=pinecone_compatible,
                            embedding=self.embeddings,
                            text_key="text",
                            namespace=""
                        )
                    except (ImportError, AttributeError):
                        # If PineconeCompatibleIndex isn't available, try manual monkey patching
                        logger.info("Attempting manual compatibility fix for Pinecone")
                        
                        # Get the Pinecone index
                        pc_index = self.pc.Index(self.index_name)
                        
                        # Monkey patch the class to make it compatible with what LangChain expects
                        # This is a fallback if the PineconeCompatibleIndex isn't available
                        import types
                        original_class = pc_index.__class__
                        pc_index.__class__ = type(
                            'PineconeIndexCompat', 
                            (original_class,), 
                            {'__class__': type('pinecone.Index', (), {})}
                        )
                        
                        # Use the monkey-patched index
                        return LangchainPinecone.from_existing_index(
                            index_name=self.index_name,
                            embedding=self.embeddings,
                            text_key="text"
                        )
                        
                except Exception as compat_error:
                    logger.error(f"Compatibility fix failed: {str(compat_error)}")
                    
                    # Last resort: try direct import creation
                    logger.info("Attempting direct vectorstore creation")
                    from langchain_community.vectorstores import Pinecone as LangChainPinecone
                    
                    # Try to create from scratch, sometimes this works when from_existing_index fails
                    return LangChainPinecone(
                        index_name=self.index_name,
                        embedding=self.embeddings,
                        text_key="text"
                    )
            else:
                # If index doesn't exist, try to create it
                logger.warning(f"Index {self.index_name} not found. Attempting to create it...")
                try:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=768,
                        metric='cosine',
                        spec={
                            "serverless": {
                                "cloud": "aws",
                                "region": "us-east-1"
                            }
                        }
                    )
                    logger.info(f"Index {self.index_name} created successfully")
                except Exception as create_error:
                    if "409" in str(create_error) or "ALREADY_EXISTS" in str(create_error):
                        logger.info(f"Index {self.index_name} already exists (or was just created)")
                    else:
                        logger.error(f"Failed to create index: {str(create_error)}")
                        raise
                
                # Try loading again with the same compatibility approach
                logger.info(f"Loading index after creation: {self.index_name}")
                return self.load_index()  # Recursive call to use the same loading logic
                
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            
            # This is a severe error, but let's provide more details for debugging
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise