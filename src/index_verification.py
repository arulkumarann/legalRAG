import os
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

def test_direct_pinecone_access():
    """Test direct access to Pinecone index"""
    try:
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "rag-xrvizion")
        
        if not api_key:
            logger.error("Missing PINECONE_API_KEY")
            return False
            
        logger.info(f"Testing direct access to Pinecone index: {index_name}")
        
        pc = Pinecone(api_key=api_key)
        
        logger.info("Listing indexes...")
        indexes = pc.list_indexes()
        logger.info(f"Index list type: {type(indexes)}")
        logger.info(f"Indexes: {indexes}")
        
        logger.info(f"Directly accessing index: {index_name}")
        try:
            index = pc.Index(index_name)
            logger.info(f"Successfully accessed index: {index}")
            
            stats = index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
            return True
        except Exception as e:
            logger.error(f"Error accessing index directly: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        return False

def test_langchain_integration():
    """Test LangChain integration with Pinecone"""
    try:
        from langchain_community.vectorstores import Pinecone as LangchainPinecone
        from langchain.embeddings.fake import FakeEmbeddings
        
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "rag-xrvizion")
        
        if not api_key:
            logger.error("Missing PINECONE_API_KEY")
            return False
            
        logger.info(f"Testing LangChain integration with index: {index_name}")
        
        embeddings = FakeEmbeddings(size=768)
        
        try:
            logger.info("Creating LangChain connection to Pinecone...")
            vectorstore = LangchainPinecone.from_existing_index(
                index_name=index_name,
                embedding=embeddings,
            )
            logger.info(f"Successfully connected via LangChain: {vectorstore}")
            return True
        except Exception as e:
            logger.error(f"Error connecting via LangChain: {str(e)}")
            return False
            
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("=== Pinecone Direct Access Test ===")
    
    direct_success = test_direct_pinecone_access()
    logger.info(f"Direct access test {'PASSED' if direct_success else 'FAILED'}")
    
    langchain_success = test_langchain_integration()
    logger.info(f"LangChain integration test {'PASSED' if langchain_success else 'FAILED'}")
    
    if direct_success and langchain_success:
        logger.info("All tests passed successfully!")
    else:
        logger.error("One or more tests failed.")