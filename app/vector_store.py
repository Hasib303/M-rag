from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from typing import List, Optional
import os
import time

class VectorStoreManager:
    def __init__(self, pinecone_api_key: str, index_name: str, openai_api_key: str = None):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        
        # Initialize Pinecone with new API
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        if openai_api_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:
            self.embeddings = OpenAIEmbeddings()
        
        self._ensure_index_exists()
        
        # Create vector store using the new pattern
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
        )
    
    def _ensure_index_exists(self):
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            print("Waiting for index to be ready...")
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print("Index is ready!")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        return self.vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4):
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    @classmethod
    def from_documents(cls, documents: List[Document], embeddings, pinecone_api_key: str, index_name: str):
        return PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )