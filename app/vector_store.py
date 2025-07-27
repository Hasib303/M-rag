from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Optional
import os
import time
from pinecone import Pinecone, ServerlessSpec

class VectorStoreManager:
    def __init__(self, pinecone_api_key: str, index_name: str, openai_api_key: str = None):
        """
        Initializes the VectorStoreManager.

        Args:
            pinecone_api_key (str): The API key for Pinecone.
            index_name (str): The name of the Pinecone index.
            openai_api_key (str, optional): The API key for OpenAI. Defaults to None.
        """
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name

        if openai_api_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:
            self.embeddings = OpenAIEmbeddings()

        self._ensure_index_exists()

        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )

    def _ensure_index_exists(self):
        """
        Ensures that the Pinecone index exists, creating it if necessary.
        """
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating a new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # Dimension for text-embedding-3-large
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
            # Wait for the index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            print("Index created successfully.")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Adds documents to the Pinecone vector store.

        Args:
            documents (List[Document]): A list of documents to add.

        Returns:
            List[str]: A list of document IDs that were added.
        """
        return self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a similarity search on the Pinecone index.

        Args:
            query (str): The query string.
            k (int, optional): The number of similar documents to return. Defaults to 4.

        Returns:
            List[Document]: A list of similar documents.
        """
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        """
        Performs a similarity search and returns documents with their scores.

        Args:
            query (str): The query string.
            k (int, optional): The number of similar documents to return. Defaults to 4.

        Returns:
            List[tuple[Document, float]]: A list of tuples, where each tuple contains a document and its similarity score.
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    @classmethod
    def from_documents(cls, documents: List[Document], embeddings, pinecone_api_key: str, index_name: str):
        """
        Creates a vector store from a list of documents.

        Args:
            documents (List[Document]): The list of documents.
            embeddings: The embedding function.
            pinecone_api_key (str): The Pinecone API key.
            index_name (str): The name of the index.

        Returns:
            PineconeVectorStore: An instance of the Pinecone vector store.
        """
        # This class method might need to be adjusted based on how you want to handle index creation
        # For now, it assumes the index is created and managed by the instance methods.
        return PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )