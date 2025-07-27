from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from typing import List
import os
import openai
from .text_chunker import TextChunker

class EmbeddingService:
    def __init__(self, api_key: str = None):
        if api_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        else:
            self.embeddings = OpenAIEmbeddings()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
    
    def embed_text(self, text):
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding

def embed_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 50, api_key: str = None) -> List[List[float]]:
    """
    Chunks the input text and returns embeddings for each chunk.
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_text(text)
    service = EmbeddingService(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    return service.embed_documents(chunks)