from .pdf_extractor import PDFExtractor
from .text_chunker import TextChunker
from .vector_store import VectorStoreManager
from .query_processor import QueryProcessor
from .embeddings import EmbeddingService
from langchain.schema import Document
from typing import List
import os

class MultilingualRAGSystem:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_index: str, retriever_k: int = 5):
        self.pdf_extractor = PDFExtractor()
        self.text_chunker = TextChunker()
        self.embedding_service = EmbeddingService(openai_api_key)
        self.vector_store = VectorStoreManager(pinecone_api_key, pinecone_index, openai_api_key)
        self.query_processor = QueryProcessor(openai_api_key)
        self.retriever_k = retriever_k
    
    def process_pdf(self, pdf_path: str) -> List[str]:
        documents = self.pdf_extractor.extract_text_from_pdf(pdf_path)
        
        # Save extracted text to file
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write("=== EXTRACTED TEXT FROM PDF ===\n\n")
            for i, doc in enumerate(documents):
                f.write(f"--- Page {i+1} ---\n")
                f.write(doc.page_content)
                f.write(f"\n\n{'='*50}\n\n")
        
        chunks = self.text_chunker.chunk_documents(documents)
        chunk_ids = self.vector_store.add_documents(chunks)
        
        print(f"Processed {len(chunks)} chunks from PDF: {pdf_path}")
        print("Extracted text saved to: extracted_text.txt")
        return chunk_ids
    
    def query(self, question: str, k: int = None) -> str:
        if k is None:
            k = self.retriever_k
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        answer = self.query_processor.process_query(question, retrieved_docs)
        return answer
    
    def query_with_sources(self, question: str, k: int = None) -> dict:
        if k is None:
            k = self.retriever_k
        retrieved_docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
        retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]
        
        answer = self.query_processor.process_query(question, retrieved_docs)
        
        sources = []
        for doc, score in retrieved_docs_with_scores:
            sources.append({
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "similarity_score": score
            })
        
        return {
            "answer": answer,
            "sources": sources
        }