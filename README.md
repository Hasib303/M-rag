# Multilingual RAG System 🚀

A comprehensive Retrieval-Augmented Generation (RAG) system designed for Bengali and English text processing. This system combines OCR-based PDF text extraction, intelligent text chunking, vector embeddings, and GPT-4 to provide accurate question-answering for multilingual documents.

## ✨ Features

- **🌐 Multilingual Support**: Seamless Bengali and English text processing
- **📄 OCR-based PDF Extraction**: Advanced text extraction using EasyOCR for superior Bengali text recognition
- **🧩 Intelligent Text Chunking**: Smart document segmentation with configurable overlap for optimal context preservation
- **🔍 Vector Database**: Pinecone integration for lightning-fast semantic search and retrieval
- **🤖 OpenAI Integration**: GPT-4 powered query processing with context-aware responses
- **🌐 REST API**: Comprehensive FastAPI-based endpoints for all functionality
- **🏗️ Modular Architecture**: Clean separation of concerns with individual service modules
- **⚡ Real-time Processing**: Instant PDF processing and query responses
- **📊 Source Attribution**: Query responses with source document references and similarity scores

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

### Environment Variables

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file with your API keys:**
   ```env
   # Required Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX=multilingual-rag
   
   # Optional Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=50
   RETRIEVER_K=5
   USE_OCR=true
   ```

### API Keys Setup

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Pinecone API Key**: Get from [Pinecone Console](https://app.pinecone.io/)

> ⚠️ **Important**: Never commit your `.env` file to version control. It contains sensitive API keys.

## 🚦 Usage

### Quick Start

1. **Start the API server**
   ```bash
   # Option 1: Using uvicorn directly
   uvicorn api:app --reload
   
   # Option 2: Using Python script
   python api.py
   ```

2. **Access the interactive documentation**
   - **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
   - **API Root**: [http://localhost:8000/](http://localhost:8000/)

3. **Process a PDF document**
   ```bash
   curl -X POST "http://localhost:8000/api/rag-system/process-pdf" \
     -H "Content-Type: application/json" \
     -d '{"pdf_path": "data/HSC26-Bangla1st-Paper.pdf"}'
   ```

4. **Ask questions (Bengali)**
   ```bash
   # Question about character description
   curl -X POST "http://localhost:8000/api/chatbot/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
   # Expected Answer: শুম্ভুনাথ
   
   # Question about character age
   curl -X POST "http://localhost:8000/api/chatbot/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"}'
   # Expected Answer: ১৫ বছর
   
   # Question about relationships
   curl -X POST "http://localhost:8000/api/chatbot/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"}'
   # Expected Answer: মামাকে
   ```

5. **Ask a question (English)**
   ```bash
   curl -X POST "http://localhost:8000/api/chatbot/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main theme of the text?"}'
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
├── app/                           # Main application package
│   ├── __init__.py               # Package initialization
│   ├── embeddings.py             # 🔗 OpenAI embeddings service
│   ├── pdf_extractor.py          # 📄 OCR-based PDF text extraction
│   ├── query_processor.py        # 🤖 GPT-4 query processing with Bengali optimization
│   ├── rag_system.py            # 🧠 Main RAG orchestrator
│   ├── text_chunker.py          # ✂️ Intelligent text chunking utilities
│   ├── vector_store.py          # 🗄️ Pinecone vector database management
│   └── routes/                   # API routing
│       ├── __init__.py
│       └── all_routes.py        # 🌐 All REST API endpoints
├── data/                         # Sample data directory
│   └── HSC26-Bangla1st-Paper.pdf # 📚 Sample Bengali textbook
├── venv/                         # Virtual environment (auto-generated)
├── api.py                        # 🚀 FastAPI application entry point
├── main.py                       # 💻 Command-line interface
├── requirements.txt              # 📦 Python dependencies
├── .env.example                  # 🔐 Environment variables template
├── .gitignore                    # 🚫 Git ignore patterns
└── README.md                     # 📖 Project documentation
```

### Core Components

| Component | Description | Technology |
|-----------|-------------|------------|
| **PDF Extractor** | Converts PDFs to text using OCR | pdf2image + EasyOCR |
| **Text Chunker** | Splits documents into manageable chunks | LangChain TextSplitter |
| **Embeddings** | Generates vector representations | OpenAI text-embedding-ada-002 |
| **Vector Store** | Stores and retrieves document vectors | Pinecone |
| **Query Processor** | Processes queries with context | OpenAI GPT-4 |
| **RAG System** | Orchestrates the entire pipeline | Custom implementation |

## 📖 Examples

### Processing Bengali Questions

```python
import requests

# Process a PDF
response = requests.post(
    "http://localhost:8000/api/rag-system/process-pdf",
    json={"pdf_path": "data/HSC26-Bangla1st-Paper.pdf"}
)

# Ask Bengali questions with expected answers
questions = [
    {
        "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "expected": "শুম্ভুনাথ"
    },
    {
        "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "expected": "১৫ বছর"
    },
    {
        "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "expected": "মামাকে"
    }
]

for q in questions:
    response = requests.post(
        "http://localhost:8000/api/chatbot/query",
        json={"question": q["question"]}
    )
    result = response.json()
    print(f"Question: {q['question']}")
    print(f"Expected: {q['expected']}")
    print(f"Got: {result['answer']}")
    print("-" * 50)

# Example Output:
# Question: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
# Expected: ১৫ বছর
# Got: ১৫ বছর
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

## 🎯 Sample Queries & Expected Results

The system has been optimized for Bengali literature analysis. Here are sample queries that demonstrate the system's capabilities:

### Character Analysis
| Question (Bengali) | Expected Answer | Question Type |
|-------------------|-----------------|---------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ | Character Description |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে | Relationship |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর | Factual Information |

### Testing Your Setup
```bash
# Test the system with these sample queries
echo "Testing Character Description Query..."
curl -X POST "http://localhost:8000/api/chatbot/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'

echo "Testing Age Information Query..."
curl -X POST "http://localhost:8000/api/chatbot/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"}'

echo "Testing Relationship Query..."
curl -X POST "http://localhost:8000/api/chatbot/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"}'
```

### Query Patterns
The system is optimized for these Bengali question patterns:
- **কাকে/কে** (Who) - Returns person names
- **কত/কতটা** (How much/many) - Returns numerical values with units
- **কোথায়** (Where) - Returns location names
- **কখন** (When) - Returns time information
- **কী/কি** (What) - Returns objects or concepts

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

#### 1. **Pinecone Connection Errors**
```bash
# Error: module 'pinecone' has no attribute 'init'
```
**Solutions:**
- Install correct Pinecone version: `pip install pinecone-client`
- Verify API key in `.env` file
- Check index name exists in Pinecone console
- Ensure network connectivity

#### 2. **OCR Extraction Issues**
```bash
# Error: pdf2image or EasyOCR not working
```
**Solutions:**
- **macOS**: `brew install poppler`
- **Ubuntu**: `sudo apt-get install poppler-utils`
- **Windows**: Download poppler and add to PATH
- Check PDF file permissions
- Verify EasyOCR model downloads (requires internet)

#### 3. **OpenAI API Errors**
```bash
# Error: Invalid API key or quota exceeded
```
**Solutions:**
- Verify OpenAI API key in `.env`
- Check API quota and billing
- Ensure correct model access (GPT-4)

#### 4. **Memory Issues with Large PDFs**
```bash
# Error: Out of memory during processing
```
**Solutions:**
- Process PDFs in smaller batches
- Reduce `CHUNK_SIZE` in `.env`
- Monitor system resources
- Use cloud deployment for large files

#### 5. **Import Errors**
```bash
# Error: ModuleNotFoundError
```
**Solutions:**
- Activate virtual environment: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Check Python version (3.8+)

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

### Planned Features
- [ ] **🌍 Multi-language Support**: Hindi, Urdu, Arabic text processing
- [ ] **📊 Advanced Analytics**: Usage metrics and performance dashboards
- [ ] **🔐 Authentication**: JWT-based user authentication and authorization
- [ ] **📝 Document Types**: Support for DOCX, TXT, and web scraping
- [ ] **🔄 Real-time Updates**: Live document synchronization
- [ ] **🎯 Custom Models**: Fine-tuned embeddings for specific domains
- [ ] **📱 Mobile App**: React Native or Flutter mobile interface
- [ ] **🐳 Docker Support**: Containerized deployment options
- [ ] **☁️ Cloud Deployment**: AWS/GCP deployment templates
- [ ] **🔍 Advanced Search**: Fuzzy search and semantic filtering

### Version Roadmap
- **v1.1**: Authentication & user management
- **v1.2**: Additional language support
- **v2.0**: Custom model training capabilities

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