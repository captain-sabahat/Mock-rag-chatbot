# Architecture docs

# User Chatbot RAG Backend - Architecture Overview

## Project Structure

\`\`\`
user-chatbot-backend/
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Shared utilities (cache keys, logging, text)
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py             # Settings loaded from .env (Pydantic BaseSettings)
│   │   ├── constants.py            # Immutable constants (API version, latency targets)
│   │   ├── model_config.py         # Model-specific tuning parameters
│   │   └── cache_config.py         # Cache configuration (keys, TTLs, strategy)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app factory, startup/shutdown
│   │   ├── routes.py               # HTTP route handlers (POST /api/v1/query)
│   │   ├── asgi.py                 # ASGI entry point for production servers
│   │   └── dependencies.py         # FastAPI dependency injection (DI)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py           # Custom exception hierarchy
│   │   ├── circuit_breaker.py      # Circuit breaker resilience pattern
│   │   ├── redis_handler.py        # Redis async client (ultra-low-latency cache)
│   │   ├── embeddings_handler.py   # Embeddings generation for semantic search
│   │   ├── vector_db_handler.py    # Qdrant vector DB async client
│   │   ├── llm_handler.py          # Fine-tuned LLM inference (answer generation)
│   │   ├── slm_handler.py          # Small LLM (document summarization)
│   │   ├── doc_ingestion.py        # Multi-format document parsing (PDF/DOCX/TXT)
│   │   └── state_manager.py        # Session state management (in-memory)
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # ILLMProvider interface
│   │   │   ├── gemini.py           # GeminiProvider
│   │   │   ├── openai.py           # OpenAIProvider
│   │   │   ├── anthropic.py        # AnthropicProvider
│   │   │   └── huggingface.py      # HuggingFaceProvider
│   │   ├── slm/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # ISLMProvider interface
│   │   │   ├── hf_causal.py        # HFCausalProvider (Phi-3)
│   │   │   └── hf_seq2seq.py       # HFSeq2SeqProvider (Distilbart)
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # IEmbeddingsProvider interface
│   │   │   └── huggingface.py      # HFEmbeddingsProvider
│   │   └── vectordb/
│   │       ├── __init__.py
│   │       ├── base.py             # IVectorDBProvider interface
│   │       ├── qdrant.py           # QdrantProvider
│   │       └── faiss.py            # FAISSProvider
│   │
│   ├── container/
│   │   ├── __init__.py
│   │   └── service_container.py    # ServiceContainer (main entry point)
│   │
│   └── pipelines/
│       ├── __init__.py
│       ├── base.py                 # Abstract base pipeline class
│       ├── schemas.py              # Pydantic request/response models
│       ├── logic_router.py         # Routing logic (LOGIC_1/2/3/4 selection)
│       ├── doc_summary.py          # LINE 1: Document summarization
│       ├── rag.py                  # RAG pipeline (semantic search + LLM)
│       ├── redis_cache.py          # LOGIC_1: Cache-only path (<10ms)
│       └── orchestrator.py         # Main coordinator (routes + executes pipelines)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures and test configuration
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_schemas.py
│   │   └── test_utils.py
│   └── integration/
│       ├── test_api_query.py
│       └── test_pipeline.py
│
├── .env.example                    # Template for environment variables
├── .gitignore                      # Git ignore rules
├── .pre-commit-config.yaml         # Pre-commit hooks (formatting, linting)
├── ARCHITECTURE.md                 # This file - architecture documentation
├── pyproject.toml                  # Python project metadata and dependencies (Modern)
├── requirements.txt                # Python dependencies (pip format)
├── requirements-dev.txt            # Development-only dependencies (testing, linting)
├── setup.py                        # Traditional setup for pip install
└── README.md                       # Project overview and getting started

\`\`\`

## Architecture Layers

### 1. API Layer (src/api/)
- **Purpose**: HTTP endpoints, request validation, dependency injection
- **Entry Point**: POST /api/v1/query
- **Responsibility**: Receive requests, inject dependencies, return responses
- **Latency**: <1ms (just routing, no computation)

### 2. Orchestration Layer (src/pipelines/)
- **Purpose**: Route requests through appropriate pipeline
- **Decision Logic**: Select LOGIC_1/2/3/4 based on cache + document flags
- **Responsibility**: Coordinate pipeline execution
- **Latency**: <5ms (routing only) + pipeline execution

### 3. Core Handlers Layer (src/core/)
- **Purpose**: Backend service interactions (Redis, Vector DB, LLM, etc.)
- **Components**: 
  - Redis: Ultra-fast caching (<10ms)
  - Vector DB: Semantic search (200-400ms)
  - LLM: Answer generation (1-2s)
  - SLM: Document summarization (2-5s)
  - Embeddings: Query encoding (100-500ms)
- **Resilience**: Circuit breaker, timeout protection, retry logic

### 4. Configuration Layer (src/config/)
- **Purpose**: Centralized settings and constants
- **Components**:
  - settings.py: Environment-based configuration (Pydantic)
  - constants.py: Immutable constants (latency targets, error codes)
  - model_config.py: Model-specific tuning
  - cache_config.py: Cache strategy and TTLs

### 5. Model Layer (src/models.py)
- **Purpose**: Model management and inference
- **Features**: Device abstraction (CPU/GPU/CUDA), quantization, async inference

## Logic Paths (4 Scenarios)

### LOGIC_1: Redis Cache HIT
- **Triggered**: User wants cache (redis_lookup=YES) + Answer exists in Redis
- **Flow**: Check Redis → Return cached answer
- **Latency Target**: <10ms
- **Use Case**: Repeated queries from same user

### LOGIC_2: Pure RAG
- **Triggered**: Cache miss (or disabled) + No document attached
- **Flow**: Embed query → Search Vector DB → LLM generates answer → Cache result
- **Latency Target**: 1-2s
- **Use Case**: Quick questions without document context

### LOGIC_3: Cache + Document (with summary)
- **Triggered**: Document attached + Cache enabled
- **Flow**: Parse doc → Summarize (SLM) → Embed → Search → Generate → Cache
- **Latency Target**: 3-4s
- **Use Case**: Document analysis with query context

### LOGIC_4: RAG + Document
- **Triggered**: Document attached + Cache miss/disabled
- **Flow**: Parse doc → Summarize → Embed query+context → Search → Generate → Cache
- **Latency Target**: 4-5s
- **Use Case**: Full document analysis

## Data Flow
USER QUERY
↓
API Route Handler
↓
Orchestrator.execute()
├─ LogicRouter: Decide LOGIC_1/2/3/4
├─ Execute selected pipeline
│ ├─ LOGIC_1: RedisCachePipeline
│ ├─ LOGIC_2: RAGPipeline (pure)
│ ├─ LOGIC_3: DocSummary + RAG
│ └─ LOGIC_4: DocSummary + RAG
└─ Return RAGResponse
↓
Cache Result (Redis)
↓
Return to User


## Key Technologies

- **Framework**: FastAPI (async, type hints, automatic docs)
- **Cache**: Redis (ultra-low-latency, connection pooling)
- **Vector DB**: Qdrant (semantic search, async client)
- **Embeddings**: SentenceTransformers (sentence-level embeddings)
- **LLM**: GPT-3.5-turbo or compatible (inference)
- **SLM**: DistilBART (fast document summarization)
- **Concurrency**: asyncio (async/await, non-blocking I/O)
- **Validation**: Pydantic (runtime type checking)

## Performance Characteristics

| Component | Latency | Notes |
|-----------|---------|-------|
| Redis GET (cache HIT) | <10ms | Ultra-fast path (LOGIC_1) |
| Embeddings | 100-500ms | GPU: 10-50ms, CPU: 100-500ms |
| Vector DB Search | 200-400ms | Semantic similarity search |
| LLM Inference | 1-2s | Answer generation with context |
| SLM Inference | 2-5s | Document summarization (with INT8: 50% faster) |
| Document Parsing | 10-500ms | Format-dependent (TXT<PDF<DOCX) |

## Scaling Considerations

### Horizontal Scaling
- Multiple server instances behind load balancer
- Shared Redis (managed cache)
- Shared Qdrant instance (vector DB)
- Stateless API (no session affinity needed)

### Vertical Scaling
- GPU support (10x faster embeddings + LLM)
- Model quantization (INT8: 50% memory reduction)
- Connection pooling (50 Redis connections)
- Batch processing (32 embeddings per batch)

### Caching Strategy
- Query cache: 1 hour TTL (70% hit rate target)
- Document summaries: 7 days TTL (cache doc processing)
- Session cache: 24 hours TTL (user context)

## Error Handling & Resilience

### Circuit Breaker Pattern
- **Open**: Service down, fail fast
- **Half-Open**: Test recovery after timeout
- **Closed**: Service working, normal operation

### Retry Logic
- Exponential backoff (2.0x factor)
- Max 3 attempts
- Only for RecoverableException

### Graceful Degradation
- Cache unavailable → Skip cache, use RAG
- Vector DB timeout → Fail with error message
- LLM timeout → Return partial results
- Document parsing error → Return error with details

## Future Enhancements (Phase 2+)

### Observability
- OpenTelemetry integration
- Prometheus metrics collection
- Distributed tracing (correlation IDs)

### Advanced Features
- Streaming responses (WebSocket)
- A/B testing (model variants)
- User feedback loop (rating answers)
- Dynamic routing (ML-based path selection)

### Optimization
- Cache warming (pre-populate common queries)
- Model pooling (multiple instances for parallelism)
- Hybrid search (keyword + semantic)
- Multi-stage retrieval (ranking, re-ranking)

### Infrastructure
- Multi-region deployment
- Redis replication (HA)
- Qdrant clustering
- Managed services (AWS, GCP, Azure)

## Development Workflow

1. **Clone & Setup**
   \`\`\`bash
   git clone repo
   cd user-chatbot-backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   pip install -r requirements-dev.txt
   cp .env.example .env
   \`\`\`

2. **Development**
   \`\`\`bash
   # Start Redis & Qdrant first
   docker-compose up -d redis qdrant
   
   # Run app with auto-reload
   uvicorn src.asgi:app --reload --port 8001
   
   # Run tests
   pytest
   
   # Format code
   black src/ tests/
   isort src/ tests/
   \`\`\`

3. **Pre-Commit**
   \`\`\`bash
   pre-commit install
   git add file.py
   git commit -m "message"  # Runs checks automatically
   \`\`\`

4. **Deployment**
   \`\`\`bash
   # Build Docker image
   docker build -t user-chatbot .
   
   # Run production
   gunicorn -w 4 -b 0.0.0.0:8001 src.asgi:app
   \`\`\`
\`\`\`

.env → Settings → Factories → Providers → Container → Routes
      (config)  (instantiate)  (isolated)  (inject)   (unaware)

