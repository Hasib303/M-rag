from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from langchain.schema import Document

# Import all modules
from ..rag_system import MultilingualRAGSystem
from ..embeddings import EmbeddingService
from ..pdf_extractor import PDFExtractor
from ..query_processor import QueryProcessor
from ..text_chunker import TextChunker
from ..vector_store import VectorStoreManager

load_dotenv()

router = APIRouter()

# Initialize all services
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX", "multilingual-rag")

if not openai_api_key or not pinecone_api_key:
    raise ValueError("Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables")

# Initialize services
rag_system = MultilingualRAGSystem(openai_api_key, pinecone_api_key, pinecone_index, retriever_k=5)
embedding_service = EmbeddingService(openai_api_key)
pdf_extractor = PDFExtractor(use_ocr=True)
query_processor = QueryProcessor(openai_api_key)
text_chunker = TextChunker()
vector_store = VectorStoreManager(pinecone_api_key, pinecone_index, openai_api_key)

# ===============================
# REQUEST/RESPONSE MODELS
# ===============================

# Chatbot Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Embeddings Models
class EmbedTextRequest(BaseModel):
    text: str

class EmbedTextsRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]

# PDF Extractor Models
class ExtractPDFRequest(BaseModel):
    pdf_path: str

class ExtractedPage(BaseModel):
    page_number: int
    content: str
    metadata: dict

class PDFExtractionResponse(BaseModel):
    total_pages: int
    pages: List[ExtractedPage]

class PDFTextResponse(BaseModel):
    text: str

# Query Processor Models
class ProcessQueryRequest(BaseModel):
    query: str
    context_documents: List[dict]

class QueryProcessResponse(BaseModel):
    answer: str

# Text Chunker Models
class ChunkTextRequest(BaseModel):
    text: str
    chunk_size: int = 1000
    chunk_overlap: int = 50

class ChunkDocumentsRequest(BaseModel):
    documents: List[dict]
    chunk_size: int = 1000
    chunk_overlap: int = 50

class TextChunksResponse(BaseModel):
    chunks: List[str]
    total_chunks: int

class DocumentChunksResponse(BaseModel):
    chunks: List[dict]
    total_chunks: int

# Vector Store Models
class AddDocumentsRequest(BaseModel):
    documents: List[dict]

class SearchRequest(BaseModel):
    query: str
    k: int = 4

class AddDocumentsResponse(BaseModel):
    message: str
    document_ids: List[str]

class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int

# RAG System Models
class ProcessPDFRequest(BaseModel):
    pdf_path: str

class QueryWithSourcesRequest(BaseModel):
    question: str
    k: int = 5

class ProcessPDFResponse(BaseModel):
    message: str
    chunks_processed: int

class SourceInfo(BaseModel):
    content: str
    metadata: dict
    similarity_score: float

class QueryWithSourcesResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]

# ===============================
# CHATBOT ENDPOINTS
# ===============================

@router.post("/chatbot/query", response_model=QueryResponse, tags=["Chatbot"])
async def chatbot_query(request: QueryRequest):
    """Ask a question and get an answer from the RAG system"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        answer = rag_system.query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# ===============================
# EMBEDDINGS ENDPOINTS
# ===============================

@router.post("/embeddings/embed-text", response_model=EmbeddingResponse, tags=["Embeddings"])
async def embed_text(request: EmbedTextRequest):
    """Generate embedding for a single text"""
    try:
        embedding = embedding_service.embed_query(request.text)
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@router.post("/embeddings/embed-texts", response_model=EmbeddingsResponse, tags=["Embeddings"])
async def embed_texts(request: EmbedTextsRequest):
    """Generate embeddings for multiple texts"""
    try:
        embeddings = embedding_service.embed_documents(request.texts)
        return EmbeddingsResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

# ===============================
# PDF EXTRACTOR ENDPOINTS
# ===============================

@router.post("/pdf-extractor/extract-pdf", response_model=PDFExtractionResponse, tags=["PDF Extractor"])
async def extract_pdf(request: ExtractPDFRequest):
    """Extract text from PDF with page-by-page breakdown"""
    try:
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        documents = pdf_extractor.extract_text_from_pdf(request.pdf_path)
        
        pages = []
        for doc in documents:
            pages.append(ExtractedPage(
                page_number=doc.metadata.get("page", 0),
                content=doc.page_content,
                metadata=doc.metadata
            ))
        
        return PDFExtractionResponse(total_pages=len(documents), pages=pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF: {str(e)}")

@router.post("/pdf-extractor/extract-pdf-text", response_model=PDFTextResponse, tags=["PDF Extractor"])
async def extract_pdf_text(request: ExtractPDFRequest):
    """Extract all text from PDF as a single string"""
    try:
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        text = pdf_extractor.extract_text_as_string(request.pdf_path)
        return PDFTextResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF text: {str(e)}")

# ===============================
# QUERY PROCESSOR ENDPOINTS
# ===============================

@router.post("/query-processor/process-query", response_model=QueryProcessResponse, tags=["Query Processor"])
async def process_query(request: ProcessQueryRequest):
    """Process a query with provided context documents"""
    try:
        documents = []
        for doc_data in request.context_documents:
            doc = Document(
                page_content=doc_data.get("content", ""),
                metadata=doc_data.get("metadata", {})
            )
            documents.append(doc)
        
        answer = query_processor.process_query(request.query, documents)
        return QueryProcessResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/query-processor/chat-history", tags=["Query Processor"])
async def get_chat_history():
    """Get the current chat history"""
    try:
        return {"chat_history": query_processor.chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@router.delete("/query-processor/clear-history", tags=["Query Processor"])
async def clear_chat_history():
    """Clear the chat history"""
    try:
        query_processor.chat_history = []
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")

# ===============================
# TEXT CHUNKER ENDPOINTS
# ===============================

@router.post("/text-chunker/chunk-text", response_model=TextChunksResponse, tags=["Text Chunker"])
async def chunk_text(request: ChunkTextRequest):
    """Split text into chunks"""
    try:
        if request.chunk_size != 1000 or request.chunk_overlap != 50:
            chunker = TextChunker(chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap)
        else:
            chunker = text_chunker
        
        chunks = chunker.chunk_text(request.text)
        return TextChunksResponse(chunks=chunks, total_chunks=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chunking text: {str(e)}")

@router.post("/text-chunker/chunk-documents", response_model=DocumentChunksResponse, tags=["Text Chunker"])
async def chunk_documents(request: ChunkDocumentsRequest):
    """Split documents into chunks"""
    try:
        documents = []
        for doc_data in request.documents:
            doc = Document(
                page_content=doc_data.get("content", ""),
                metadata=doc_data.get("metadata", {})
            )
            documents.append(doc)
        
        if request.chunk_size != 1000 or request.chunk_overlap != 50:
            chunker = TextChunker(chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap)
        else:
            chunker = text_chunker
        
        chunked_docs = chunker.chunk_documents(documents)
        
        chunks = []
        for doc in chunked_docs:
            chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return DocumentChunksResponse(chunks=chunks, total_chunks=len(chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chunking documents: {str(e)}")

# ===============================
# VECTOR STORE ENDPOINTS
# ===============================

@router.post("/vector-store/add-documents", response_model=AddDocumentsResponse, tags=["Vector Store"])
async def add_documents(request: AddDocumentsRequest):
    """Add documents to the vector store"""
    try:
        documents = []
        for doc_data in request.documents:
            doc = Document(
                page_content=doc_data.get("content", ""),
                metadata=doc_data.get("metadata", {})
            )
            documents.append(doc)
        
        document_ids = vector_store.add_documents(documents)
        return AddDocumentsResponse(
            message=f"Successfully added {len(documents)} documents",
            document_ids=document_ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")

@router.post("/vector-store/search", response_model=SearchResponse, tags=["Vector Store"])
async def search_documents(request: SearchRequest):
    """Search for similar documents"""
    try:
        results = vector_store.similarity_search(request.query, k=request.k)
        
        search_results = []
        for doc in results:
            search_results.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata
            ))
        
        return SearchResponse(results=search_results, total_results=len(search_results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@router.post("/vector-store/search-with-scores", response_model=SearchResponse, tags=["Vector Store"])
async def search_documents_with_scores(request: SearchRequest):
    """Search for similar documents with similarity scores"""
    try:
        results = vector_store.similarity_search_with_score(request.query, k=request.k)
        
        search_results = []
        for doc, score in results:
            search_results.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score
            ))
        
        return SearchResponse(results=search_results, total_results=len(search_results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents with scores: {str(e)}")

# ===============================
# RAG SYSTEM ENDPOINTS
# ===============================

@router.post("/rag-system/process-pdf", response_model=ProcessPDFResponse, tags=["RAG System"])
async def process_pdf(request: ProcessPDFRequest):
    """Process a PDF file and add to vector database"""
    try:
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        chunk_ids = rag_system.process_pdf(request.pdf_path)
        
        return ProcessPDFResponse(
            message=f"Successfully processed PDF: {request.pdf_path}",
            chunks_processed=len(chunk_ids)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.post("/rag-system/query", response_model=QueryResponse, tags=["RAG System"])
async def rag_query(request: QueryRequest):
    """Ask a question and get an answer from the RAG system"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        answer = rag_system.query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@router.post("/rag-system/query-with-sources", response_model=QueryWithSourcesResponse, tags=["RAG System"])
async def query_with_sources(request: QueryWithSourcesRequest):
    """Ask a question and get an answer with source documents"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = rag_system.query_with_sources(request.question, k=request.k)
        
        sources = []
        for source in result["sources"]:
            sources.append(SourceInfo(
                content=source["content"],
                metadata=source["metadata"],
                similarity_score=source["similarity_score"]
            ))
        
        return QueryWithSourcesResponse(answer=result["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question with sources: {str(e)}")

@router.get("/rag-system/health", tags=["RAG System"])
async def health_check():
    """Check if the RAG system is healthy"""
    return {"status": "healthy", "message": "RAG system is operational"}