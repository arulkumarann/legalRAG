import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
    CHUNK_SIZE: int = 350
    CHUNK_OVERLAP: int = 50
    INDEX_NAME: str = "legal-rag"

    def validate(self):
        if not all([self.GROQ_API_KEY, self.PINECONE_API_KEY]):
            raise ValueError("Missing required environment variables. Please check your .env file.")

config = Config()