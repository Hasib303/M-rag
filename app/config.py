import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Configuration settings for the application, loaded from environment variables.
    """
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "multilingual-rag")
    
    # Retriever Configuration
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", 5))

    def __init__(self):
        """
        Validates that all necessary environment variables are set.
        """
        if not self.OPENAI_API_KEY:
            raise ValueError("Missing environment variable: OPENAI_API_KEY")
        if not self.PINECONE_API_KEY:
            raise ValueError("Missing environment variable: PINECONE_API_KEY")

# Instantiate settings to be imported by other modules
try:
    settings = Settings()
except ValueError as e:
    print(f"Error: {e}")
    # Exit or handle the error as appropriate for your application
    exit(1)
