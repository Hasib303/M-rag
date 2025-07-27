from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import os

from ..config import Settings, settings
from ..rag_system import MultilingualRAGSystem

router = APIRouter()

# Dependency to get the settings
def get_settings():
    return settings

# Initialize the RAG system using the settings
def get_rag_system(settings: Settings = Depends(get_settings)):
    return MultilingualRAGSystem(
        openai_api_key=settings.OPENAI_API_KEY,
        pinecone_api_key=settings.PINECONE_API_KEY,
        pinecone_index=settings.PINECONE_INDEX,
        retriever_k=settings.RETRIEVER_K
    )

# ===============================
# Pydantic Models for API I/O
# ===============================

class ProcessPDFRequest(BaseModel):
    pdf_path: str

class ProcessPDFResponse(BaseModel):
    message: str
    chunks_processed: int

class QueryRequest(BaseModel):
    question: str
    k: int = None

class SourceInfo(BaseModel):
    content: str
    metadata: dict
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]

# ===============================
# API Endpoints
# ===============================

@router.post("/process-pdf", response_model=ProcessPDFResponse, tags=["RAG System"])
async def process_pdf_endpoint(
    request: ProcessPDFRequest,
    rag_system: MultilingualRAGSystem = Depends(get_rag_system)
):
    """
    Process a PDF file, chunk it, embed it, and store it in the vector database.
    """
    pdf_path = request.pdf_path
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found at: {pdf_path}")
    
    try:
        chunk_ids = rag_system.process_pdf(pdf_path)
        return ProcessPDFResponse(
            message=f"Successfully processed PDF: {pdf_path}",
            chunks_processed=len(chunk_ids)
        )
    except Exception as e:
        # Log the exception for debugging
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the PDF.")

@router.post("/query", response_model=QueryResponse, tags=["RAG System"])
async def query_endpoint(
    request: QueryRequest,
    rag_system: MultilingualRAGSystem = Depends(get_rag_system)
):
    """
    Ask a question and get a response augmented by documents from the vector database.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        result = rag_system.query_with_sources(question, k=request.k)
        return QueryResponse(**result)
    except Exception as e:
        # Log the exception for debugging
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing the query.")

@router.get("/health", tags=["RAG System"])
async def health_check():
    """
    Check if the API is running.
    """
    return {"status": "healthy"}
