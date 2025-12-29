## FILE 5: README.md

\`\`\`markdown
# User Chatbot Backend - RAG Pipeline

Production-grade Retrieval-Augmented Generation (RAG) backend for user chatbot.

## Features

- **Ultra-Fast Caching**: <10ms cache hits via Redis
- **Semantic Search**: Vector-based document retrieval (Qdrant)
- **LLM Integration**: Context-aware answer generation
- **Document Support**: Parse PDF, DOCX, TXT
- **Auto Summarization**: Fast document summaries (SLM)
- **Async/Non-Blocking**: Full async I/O for scalability
- **Resilience**: Circuit breaker, timeout protection, graceful degradation
- **Production-Ready**: Error handling, logging, monitoring

## Quick Start

### Prerequisites
- Python 3.10+
- Redis
- Qdrant Vector DB
- GPU (optional, for faster inference)

### Setup

\`\`\`bash
# Clone repository
git clone <repo-url>
cd user-chatbot-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start services
docker-compose up -d

# Run server
uvicorn src.asgi:app --reload --port 8001
\`\`\`

### API Endpoint

**POST** `/api/v1/query`

Request:
\`\`\`json
{
  "user_id": "user123",
  "session_id": "sess_abc",
  "prompt": "What is machine learning?",
  "redis_lookup": "yes",
  "doc_attached": "no"
}
\`\`\`

Response:
\`\`\`json
{
  "status": "success",
  "result": {
    "answer": "Machine learning is...",
    "sources": [...],
    "processing_time_ms": 1234
  },
  "logic_path": "logic_2_pure_rag",
  "request_id": "req_abc123"
}
\`\`\`

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

### Key Components

- **API Layer**: FastAPI routes, dependency injection
- **Orchestrator**: Pipeline routing (LOGIC_1/2/3/4)
- **Core Handlers**: Redis, Vector DB, LLM, SLM
- **Configuration**: Settings, constants, cache config

## Performance

| Path | Latency | Description |
|------|---------|-------------|
| LOGIC_1 | <10ms | Cache hit (ultra-fast) |
| LOGIC_2 | 1-2s | Pure RAG (no document) |
| LOGIC_3 | 3-4s | Document + cache |
| LOGIC_4 | 4-5s | Document + RAG |

## Development

### Testing

\`\`\`bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/unit/test_models.py
\`\`\`

### Code Quality

\`\`\`bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking (optional)
mypy src/
\`\`\`

### Pre-Commit

\`\`\`bash
# Install hooks
pre-commit install

# Manual run
pre-commit run --all-files
\`\`\`

## Configuration

See `.env.example` for all available settings:

- **Device**: CPU/GPU/CUDA/Metal/NPU selection
- **Models**: LLM, SLM, Embeddings model names
- **Timeouts**: Per-component timeout configuration
- **Cache**: Redis URL, pool size, TTL
- **Vector DB**: Qdrant URL, top-k results, score threshold

## Deployment

### Docker

\`\`\`bash
docker build -t user-chatbot .
docker run -p 8001:8001 --env-file .env user-chatbot
\`\`\`

### Production

\`\`\`bash
gunicorn -w 4 -b 0.0.0.0:8001 src.asgi:app
\`\`\`

## Monitoring

- **Metrics**: Latency per logic path, cache hit rate, error rate
- **Logs**: Structured logging with correlation IDs
- **Alerts**: Alert on circuit breaker state changes, high latency

## Future Enhancements

- OpenTelemetry integration
- Streaming responses (WebSocket)
- A/B testing support
- User feedback loop
- Multi-region deployment

## License

[Your License Here]

## Support

For issues or questions, please create a GitHub issue or contact [email].
\`\`\`

---