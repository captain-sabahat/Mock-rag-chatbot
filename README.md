RAG Admin Pipeline
🚀 Retrieval-Augmented Generation (RAG) Administration Platform
A production-ready, enterprise-grade Retrieval-Augmented Generation (RAG) pipeline built with FastAPI, designed for document ingestion, semantic search, and AI-powered question answering.

📋 Table of Contents
Features

Architecture

Quick Start

Installation

Configuration

Usage

API Documentation

Project Structure

Technologies

Contributing

License

✨ Features
Core Capabilities
✅ Document Ingestion

Support for PDF, DOCX, TXT, Markdown files

Automatic text extraction and parsing

OCR support for scanned documents

✅ Intelligent Chunking

Semantic-aware text splitting

Configurable chunk size and overlap

Sentence and paragraph preservation

✅ Embedding Generation

BGE (BAAI General Embeddings) models

GPU acceleration support

Batch processing

Embedding caching

✅ Vector Storage & Search

FAISS (Facebook AI Similarity Search)

Million-scale vector indexing

Sub-millisecond similarity search

Metadata filtering

✅ Retrieval & Ranking

Top-K similar document retrieval

BM25 + vector hybrid search

Relevance scoring

✅ LLM Integration

OpenAI GPT-4, GPT-3.5

Anthropic Claude

Custom LLM support

✅ Admin Dashboard (API)

Document management

Knowledge base administration

Query testing

Analytics & monitoring

🏗️ Architecture
System Design
text
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT INTERFACE                        │
│              (Web UI / API Clients)                         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   API GATEWAY (FastAPI)                     │
│         (Authentication, Routing, Rate Limiting)            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────┬───────────┼───────────┬─────────────┐
│            │           │           │             │
▼            ▼           ▼           ▼             ▼
┌──────────┐ ┌────────┐ ┌────────┐ ┌───────┐  ┌──────────┐
│ Document │ │Chunking│ │Embedding│ │Vector │  │ Retrieval│
│Processing│ │ Module │ │ Module  │ │ Store │  │ Module   │
└──────────┘ └────────┘ └────────┘ └───────┘  └──────────┘
     │            │          │         │          │
     └────────────┴──────────┴─────────┴──────────┘
                         │
              ┌──────────▼─────────┐
              │    Data Layer      │
              ├────────────────────┤
              │ PostgreSQL (Meta)  │
              │ Redis (Cache)      │
              │ FAISS (Vectors)    │
              └────────────────────┘
                         │
              ┌──────────▼─────────┐
              │   LLM Services     │
              ├────────────────────┤
              │ OpenAI             │
              │ Anthropic          │
              │ Custom Models      │
              └────────────────────┘
Data Flow
text
Document Upload
    ↓
Document Parsing
    ├─ PDF extraction
    ├─ DOCX extraction
    ├─ TXT reading
    └─ Metadata extraction
    ↓
Text Cleaning & Normalization
    ├─ Remove special characters
    ├─ Normalize whitespace
    └─ Handle encoding
    ↓
Semantic Chunking
    ├─ Split by paragraphs
    ├─ Respect sentence boundaries
    └─ Apply overlap
    ↓
Embedding Generation
    ├─ BGE model inference
    ├─ Batch processing
    └─ Normalize vectors
    ↓
Vector Indexing
    ├─ Store in FAISS
    ├─ Index metadata
    └─ Cache embeddings
    ↓
Knowledge Base Ready
    └─ Ready for queries
🚀 Quick Start
Prerequisites
Python 3.9+

PostgreSQL 12+

Redis 6+

4GB RAM minimum (8GB+ recommended)

GPU optional (NVIDIA CUDA for acceleration)

1. Clone Repository
bash
git clone https://github.com/yourusername/rag-admin-pipeline.git
cd rag-admin-pipeline
2. Create Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
4. Configure Environment
bash
cp .env.example .env
# Edit .env with your actual values
nano .env
5. Initialize Database
bash
alembic upgrade head
6. Start Application
bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
7. Access Application
text
API Documentation: http://localhost:8000/docs
Alternative Docs:  http://localhost:8000/redoc
Health Check:      http://localhost:8000/health
📦 Installation
Detailed Installation Steps
1. System Dependencies (Ubuntu/Debian)
bash
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv postgresql redis-server
2. Python Environment
bash
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
3. Project Dependencies
bash
# Core dependencies
pip install -r requirements.txt

# Optional: GPU support for FAISS
pip install faiss-gpu

# Optional: PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
4. Database Setup
bash
# Create PostgreSQL database
createdb rag_admin_db

# Run migrations
alembic upgrade head

# Optional: Load sample data
python scripts/load_samples.py
5. Redis Setup
bash
# Start Redis server
redis-server

# Test connection
redis-cli ping  # Should output: PONG
⚙️ Configuration
Environment Variables (.env)
text
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/rag_admin_db

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key-min-32-chars
ALGORITHM=HS256

# API Keys
OPENAI_API_KEY=sk-your-key-here

# Vector DB
VECTORDB_DIMENSION=384
VECTORDB_INDEX_TYPE=flat

# Embeddings
EMBEDDER_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDER_DEVICE=cuda  # or cpu

# Application
DEBUG=false
ENV=production
LOG_LEVEL=INFO
Configuration Files
text
config/
├── settings.yaml          # Application settings
├── models.yaml           # Model configurations
└── logging.yaml          # Logging configuration
💻 Usage
API Examples
1. Upload Document
bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
2. Create Knowledge Base
bash
curl -X POST "http://localhost:8000/api/v1/knowledge-bases" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Knowledge Base",
    "description": "Company documentation"
  }'
3. Query Knowledge Base
bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I use this feature?",
    "knowledge_base_id": "kb_123",
    "top_k": 5
  }'
Python Client
python
from rag_client import RAGClient

# Initialize client
client = RAGClient(
    api_url="http://localhost:8000",
    api_key="your-api-key"
)

# Upload document
document_id = client.upload_document(
    file_path="document.pdf",
    kb_id="kb_123"
)

# Query
results = client.query(
    query="What is this about?",
    kb_id="kb_123",
    top_k=5
)

for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
📚 API Documentation
Interactive Documentation
Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

Key Endpoints
Method	Endpoint	Description
POST	/api/v1/documents/upload	Upload document
GET	/api/v1/documents/{id}	Get document details
DELETE	/api/v1/documents/{id}	Delete document
POST	/api/v1/query	Query knowledge base
GET	/api/v1/knowledge-bases	List knowledge bases
POST	/api/v1/knowledge-bases	Create knowledge base
📁 Project Structure
text
rag-admin-pipeline/
├── src/
│   ├── main.py                 # FastAPI application
│   ├── config/                 # Configuration
│   ├── models/                 # Database models
│   ├── schemas/                # Pydantic schemas
│   ├── api/
│   │   ├── routes/            # API endpoints
│   │   ├── auth/              # Authentication
│   │   └── middleware/        # Request middleware
│   ├── services/              # Business logic
│   ├── tools/
│   │   ├── chunking/          # Document chunking
│   │   ├── embeddings/        # Embedding generation
│   │   ├── ingestion/         # Document ingestion
│   │   ├── vectordb/          # Vector database
│   │   └── preprocessors/     # Text preprocessing
│   ├── db/                    # Database setup
│   └── utils/                 # Utility functions
│
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── config/
│   ├── settings.yaml          # Settings
│   ├── logging.yaml           # Logging config
│   └── models.yaml            # Model configs
│
├── scripts/                    # Utility scripts
│   ├── load_samples.py        # Load sample data
│   ├── create_indexes.py      # Create database indexes
│   └── migrate.py             # Database migration
│
├── docs/                       # Documentation
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── DEVELOPMENT.md
│
├── .env                        # Environment variables (NEVER commit)
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── docker-compose.yml         # Docker setup
├── Dockerfile                 # Docker image
└── alembic.ini                # Database migration config
🛠️ Technologies
Backend
FastAPI: Modern async web framework

SQLAlchemy: ORM and database toolkit

Pydantic: Data validation

AI/ML
FAISS: Vector similarity search

Sentence Transformers: BGE embeddings

LangChain: LLM orchestration

OpenAI/Anthropic: LLM APIs

Database
PostgreSQL: Primary data store

Redis: Caching and sessions

FAISS: Vector indexes

DevOps
Docker: Containerization

Docker Compose: Local development

Alembic: Database migrations

🧪 Testing
Run Tests
bash
# All tests
pytest

# With coverage
pytest --cov=src

# Specific test file
pytest tests/unit/test_chunking.py

# Verbose output
pytest -v
Test Coverage
bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
📖 Development
Install Development Dependencies
bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
Code Quality
bash
# Format code
black src tests

# Check linting
flake8 src tests

# Type checking
mypy src
Local Development
bash
# Start with auto-reload
uvicorn src.main:app --reload

# With debug logging
DEBUG=true LOG_LEVEL=DEBUG uvicorn src.main:app --reload
🚢 Deployment
Docker Deployment
bash
# Build image
docker build -t rag-pipeline:latest .

# Run container
docker run -p 8000:8000 --env-file .env rag-pipeline:latest
Docker Compose (All Services)
bash
docker-compose up -d
Production Deployment
See DEPLOYMENT.md

📊 Monitoring & Logging
Prometheus Metrics
text
http://localhost:8001/metrics
Logs Location
bash
tail -f logs/app.log
Health Check
bash
curl http://localhost:8000/health
🤝 Contributing
Development Workflow
Create feature branch: git checkout -b feature/my-feature

Make changes and commit: git commit -am "Add feature"

Push to branch: git push origin feature/my-feature

Create Pull Request

Code Standards
Follow PEP 8

100% test coverage for new features

Update documentation

Run code quality checks

📝 License
This project is licensed under the MIT License - see LICENSE file for details.

🆘 Troubleshooting
Common Issues
Issue: Database connection failed

bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Check connection string in .env
echo $DATABASE_URL
Issue: Redis connection failed

bash
# Check Redis is running
redis-cli ping

# Update REDIS_URL in .env
Issue: FAISS installation fails

bash
# Try CPU version
pip install faiss-cpu

# Or specific GPU version
pip install faiss-gpu

Getting Help

📞 Contact & Support
