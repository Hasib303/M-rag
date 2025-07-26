from langchain_openai import OpenAIEmbeddings
from typing import List
import os

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