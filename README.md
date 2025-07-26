# Multilingual RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system designed for Bengali and English text processing. This system uses OCR-based PDF text extraction, intelligent text chunking, vector embeddings, and GPT-4 for accurate question answering.

## 🚀 Features

- **Multilingual Support**: Bengali and English text processing
- **OCR-based PDF Extraction**: Advanced text extraction using EasyOCR for better Bengali text recognition
- **Intelligent Text Chunking**: Smart document segmentation with configurable overlap
- **Vector Database**: Pinecone integration for efficient semantic search
- **OpenAI Integration**: GPT-4 powered query processing with context-aware responses
- **REST API**: Comprehensive FastAPI-based endpoints for all functionality
- **Modular Architecture**: Clean separation of concerns with individual service modules

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- OpenAI API Key
- Pinecone API Key
- Virtual Environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Multilingual RAG"
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=multilingual-rag
```

## 🚦 Usage

### Quick Start

1. **Start the API server**
   ```bash
   uvicorn api:app --reload
   ```

2. **Access the interactive documentation**
   - Open: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

3. **Process a PDF document**
   ```bash
   curl -X POST "http://localhost:8000/api/rag-system/process-pdf" \
     -H "Content-Type: application/json" \
     -d '{"pdf_path": "data/HSC26-Bangla1st-Paper.pdf"}'
   ```

4. **Ask a question**
   ```bash
   curl -X POST "http://localhost:8000/api/chatbot/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
   ```

### Command Line Interface

Run the interactive CLI:
```bash
python main.py
```

## 📡 API Endpoints

All endpoints are prefixed with `/api` and organized by functionality:

### Chatbot
- `POST /api/chatbot/query` - Main question-answering endpoint

### RAG System
- `POST /api/rag-system/process-pdf` - Process PDF into vector database
- `POST /api/rag-system/query` - Query with context retrieval
- `POST /api/rag-system/query-with-sources` - Query with source attribution
- `GET /api/rag-system/health` - System health check

### PDF Processing
- `POST /api/pdf-extractor/extract-pdf` - Extract text with page breakdown
- `POST /api/pdf-extractor/extract-pdf-text` - Extract as single text string

### Text Processing
- `POST /api/text-chunker/chunk-text` - Split text into chunks
- `POST /api/text-chunker/chunk-documents` - Split documents into chunks

### Embeddings
- `POST /api/embeddings/embed-text` - Generate single text embedding
- `POST /api/embeddings/embed-texts` - Generate multiple text embeddings

### Vector Database
- `POST /api/vector-store/add-documents` - Add documents to vector store
- `POST /api/vector-store/search` - Search similar documents
- `POST /api/vector-store/search-with-scores` - Search with similarity scores

### Query Processing
- `POST /api/query-processor/process-query` - Process query with context
- `GET /api/query-processor/chat-history` - Get chat history
- `DELETE /api/query-processor/clear-history` - Clear chat history

## 📁 Project Structure

```
Multilingual RAG/
├── app/
│   ├── __init__.py
│   ├── embeddings.py          # OpenAI embeddings service
│   ├── pdf_extractor.py       # OCR-based PDF text extraction
│   ├── query_processor.py     # GPT-4 query processing
│   ├── rag_system.py         # Main RAG orchestrator
│   ├── text_chunker.py       # Text chunking utilities
│   ├── vector_store.py       # Pinecone vector database
│   └── routes/
│       ├── __init__.py
│       └── all_routes.py     # All API endpoints
├── data/
│   └── HSC26-Bangla1st-Paper.pdf  # Sample Bengali PDF
├── api.py                    # FastAPI application
├── main.py                   # CLI interface
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 📖 Examples

### Processing Bengali Questions

```python
import requests

# Process a PDF
response = requests.post(
    "http://localhost:8000/api/rag-system/process-pdf",
    json={"pdf_path": "data/HSC26-Bangla1st-Paper.pdf"}
)

# Ask a Bengali question
response = requests.post(
    "http://localhost:8000/api/chatbot/query",
    json={"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}
)

print(response.json())
# Output: {"answer": "শুম্ভুনাথ"}
```

### Embedding Generation

```python
# Generate embedding for Bengali text
response = requests.post(
    "http://localhost:8000/api/embeddings/embed-text",
    json={"text": "বাংলা ভাষা"}
)

embedding = response.json()["embedding"]  # List of 1536 floats
```

### Document Chunking

```python
# Chunk a large document
response = requests.post(
    "http://localhost:8000/api/text-chunker/chunk-text",
    json={
        "text": "Your large text document here...",
        "chunk_size": 500,
        "chunk_overlap": 100
    }
)

chunks = response.json()["chunks"]
```

## 🔧 Key Components

### PDF Extractor
- Uses `pdf2image` to convert PDF pages to images
- Employs `EasyOCR` with Bengali language support
- Fallback to PyPDF for standard text extraction

### Text Chunker
- Recursive character-based text splitting
- Configurable chunk size and overlap
- Preserves document metadata

### Vector Store
- Pinecone integration for scalable vector search
- Automatic index creation and management
- Similarity search with configurable results count

### Query Processor
- GPT-4 powered natural language processing
- Context-aware response generation
- Chat history management
- Bilingual prompt engineering

## 🎯 Use Cases

- **Educational Content Analysis**: Process Bengali textbooks and answer questions
- **Document Q&A**: Extract information from multilingual PDFs
- **Research Assistant**: Semantic search through academic papers
- **Content Management**: Organize and query large document collections

## ⚡ Performance Optimization

- **Caching**: Efficient embedding caching for repeated queries
- **Chunking Strategy**: Optimized chunk sizes for Bengali text
- **Vector Search**: Fast similarity search with Pinecone
- **OCR Processing**: Parallel page processing for large PDFs

## 🐛 Troubleshooting

### Common Issues

1. **Pinecone Connection Errors**
   - Verify API key is correct
   - Check index name exists
   - Ensure proper network connectivity

2. **OCR Extraction Issues**
   - Install system dependencies for pdf2image
   - Check PDF file permissions
   - Verify EasyOCR model downloads

3. **Memory Issues with Large PDFs**
   - Process PDFs in batches
   - Adjust chunk sizes
   - Monitor system resources

## 📊 System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB for models and dependencies
- **Network**: Stable internet for API calls
- **OS**: Linux, macOS, or Windows

## 🔒 Security Considerations

- Store API keys securely in environment variables
- Implement rate limiting for production use
- Validate file uploads and paths
- Monitor API usage and costs

## 📈 Future Enhancements

- [ ] Support for more languages
- [ ] Advanced document preprocessing
- [ ] Custom embedding models
- [ ] Real-time document updates
- [ ] Authentication and authorization
- [ ] Metrics and monitoring dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the documentation at `/docs`
- Review the API documentation at `/api/docs`

## 🙏 Acknowledgments

- OpenAI for GPT-4 and embedding models
- Pinecone for vector database services
- LangChain for RAG framework
- EasyOCR for multilingual OCR capabilities
- FastAPI for the web framework

---

**Built with ❤️ for multilingual text processing**