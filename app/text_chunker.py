from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def chunk_text(self, text: str) -> List[str]:
        chunks = self.text_splitter.split_text(text)
        return chunks