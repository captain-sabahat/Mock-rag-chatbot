"""
================================================================================
RAG ADMIN PIPELINE - Root Package Initialization
================================================================================

PURPOSE:
--------
Initialize the RAG (Retrieval-Augmented Generation) Admin Pipeline package.
Exports all public APIs at the root level for easy importing.

ARCHITECTURE:
--------------
    src/
    ├── api/             (FastAPI routes & models)
    ├── core/            (Base classes, exceptions, patterns)
    ├── cache/           (Session storage & caching)
    ├── pipeline/        (Orchestration & nodes)
    ├── utils/           (Utilities & helpers)
    └── __init__.py      (This file - root exports)

STRUCTURE:
-----------
Level 1 Imports (Direct from submodules):
  - src.core.*
  - src.cache.*
  - src.pipeline.*
  - src.utils.*
  - src.api.*

Level 2 Imports (Root level):
  - from src import BaseTool, CircuitBreakerManager
  - from src import get_orchestrator, PipelineState
  - from src import SessionStoreFactory
  - from src import validate_file, measure_time

VERSION:
--------
    1.0.0 - Initial release
    - Complete RAG pipeline
    - 5-node processing workflow
    - Vector database integration
    - Error handling & retry logic
    - Full test coverage

ENVIRONMENT VARIABLES:
----------------------
    LOG_LEVEL=INFO          # Logging level (DEBUG, INFO, WARNING, ERROR)
    ENVIRONMENT=production  # Environment (development, staging, production)
    ENABLE_METRICS=true    # Enable Prometheus metrics
    CACHE_BACKEND=redis    # Cache backend (redis, memory, disk)
    DB_URL=...             # Database connection string
    VECTORDB_BACKEND=qdrant # Vector DB (qdrant, pinecone, faiss)

QUICK START:
-----------
    # Initialize RAG pipeline
    from src import get_orchestrator, validate_file
    import uuid
    
    # Validate file
    is_valid, error = validate_file("document.pdf", file_bytes)
    if not is_valid:
        raise ValueError(error)
    
    # Process document
    orchestrator = get_orchestrator()
    request_id = await orchestrator.process_document(
        request_id=str(uuid.uuid4()),
        file_name="document.pdf",
        file_content=file_bytes
    )
    
    # Check status
    status = await orchestrator.get_status(request_id)
    print(f"Status: {status['status']}")
    print(f"Progress: {status['progress_percent']}%")

MODULES:
--------
    ✅ core/            - Base classes, exceptions, patterns
    ✅ cache/           - Session storage, caching layer
    ✅ pipeline/        - Orchestration, nodes, schemas
    ✅ utils/           - File validation, helpers
    ✅ api/             - FastAPI routes, models

TESTING:
--------
    pytest tests/           # Run all tests
    pytest tests/ -v        # Verbose output
    pytest tests/ --cov     # With coverage
    pytest -k "test_name"   # Run specific test

DEPLOYMENT:
-----------
    docker build -t rag-pipeline .
    docker run -p 8000:8000 rag-pipeline

API:
----
    POST /api/v1/upload              - Upload document
    GET  /api/v1/status/{request_id} - Check processing status
    POST /api/v1/query               - Query processed documents
    GET  /api/v1/health              - Health check

================================================================================
"""

import logging
import sys
from typing import Optional

# Set up root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Version
__version__ = "1.0.0"
__author__ = "RAG Pipeline Team"
__title__ = "RAG Admin Pipeline"

logger.info(f"🚀 Initializing {__title__} v{__version__}")


# ============================================================================
# CORE IMPORTS - Exception classes, base classes, patterns
# ============================================================================

from src.core import (
    # Base classes
    BaseTool,
    ToolConfig,
    # Exceptions
    RAGPipelineException,
    ValidationError,
    ProcessingError,
    ChunkingError,
    EmbeddingError,
    VectorDBError,
    ConfigurationError,
    # Patterns
    CircuitBreakerManager,
    CircuitBreakerConfig,
    CircuitState,
)

# ============================================================================
# CACHE IMPORTS - Session storage, state management
# ============================================================================

from src.cache import (
    # Factory
    SessionStoreFactory,
    # Base interface
    SessionStore,
    # Models
    SessionModel,
    NodeCheckpoint,
    SessionMessage,
)

# ============================================================================
# PIPELINE IMPORTS - Orchestration, state, nodes
# ============================================================================

from src.pipeline import (
    # Orchestrator
    get_orchestrator,
    PipelineOrchestrator,
    # State
    PipelineState,
    NodeStatus,
    NodeInput,
    NodeOutput,
    NodeCheckpointData,
)

from src.pipeline.nodes import (
    # Nodes
    ingestion_node,
    preprocessing_node,
    chunking_node,
    embedding_node,
    vectordb_node,
)

# ============================================================================
# UTILS IMPORTS - Validation, helpers, utilities
# ============================================================================

from src.utils import (
    # File validation
    validate_file,
    validate_file_type,
    validate_file_size,
    get_file_type,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    # Helpers
    measure_time,
    format_size,
    format_duration,
    safe_json_dumps,
    retry_on_exception,
    sanitize_filename,
)

# ============================================================================
# API IMPORTS - Routes, models, middleware
# ============================================================================

try:
    from src.api import (
        # Routers
        router as api_router,
        # Models
        UploadRequest,
        UploadResponse,
        QueryRequest,
        QueryResponse,
        StatusResponse,
        ErrorResponse,
    )
except ImportError as e:
    logger.warning(f"⚠️  API imports failed (optional): {str(e)}")


# ============================================================================
# PUBLIC API - What users can import from `src`
# ============================================================================

__all__ = [
    # Version
    "__version__",
    "__title__",
    "__author__",
    
    # Core classes & exceptions
    "BaseTool",
    "ToolConfig",
    "RAGPipelineException",
    "ValidationError",
    "ProcessingError",
    "ChunkingError",
    "EmbeddingError",
    "VectorDBError",
    "ConfigurationError",
    
    # Patterns
    "CircuitBreakerManager",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    
    # Cache
    "SessionStoreFactory",
    "RedisSessionStore",
    "MemorySessionStore",
    "SessionData",
    
    # Pipeline
    "get_orchestrator",
    "PipelineOrchestrator",
    "PipelineState",
    "NodeStatus",
    "NodeInput",
    "NodeOutput",
    "NodeCheckpointData",
    
    # Nodes
    "ingestion_node",
    "preprocessing_node",
    "chunking_node",
    "embedding_node",
    "vectordb_node",
    
    # Utils
    "validate_file",
    "validate_file_type",
    "validate_file_size",
    "get_file_type",
    "ALLOWED_EXTENSIONS",
    "MAX_FILE_SIZE",
    "measure_time",
    "format_size",
    "format_duration",
    "safe_json_dumps",
    "retry_on_exception",
    "sanitize_filename",
    
    # API (optional)
    "api_router",
    "UploadRequest",
    "UploadResponse",
    "QueryRequest",
    "QueryResponse",
    "StatusResponse",
    "ErrorResponse",
]


# ============================================================================
# INITIALIZATION HELPERS
# ============================================================================

def initialize_pipeline(
    log_level: str = "INFO",
    cache_backend: str = "memory",
    vectordb_backend: str = "qdrant",
) -> None:
    """
    Initialize the RAG pipeline with configuration.
    
    Call this once at application startup.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        cache_backend: Cache backend (memory, redis)
        vectordb_backend: Vector DB backend (qdrant, pinecone, faiss)
        
    Example:
        from src import initialize_pipeline
        
        initialize_pipeline(
            log_level="INFO",
            cache_backend="redis",
            vectordb_backend="qdrant"
        )
    """
    # Set logging level
    log_level = log_level.upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    logger.info(f"📋 Logging level set to {log_level}")
    
    # Initialize cache
    from src.cache import SessionStoreFactory
    SessionStoreFactory.set_backend(cache_backend)
    logger.info(f"💾 Cache backend set to {cache_backend}")
    
    # Initialize orchestrator
    from src.pipeline import get_orchestrator
    orchestrator = get_orchestrator()
    logger.info(f"🚀 Pipeline orchestrator initialized")
    
    # Health check
    health = orchestrator.health_check()
    logger.info(f"✅ Pipeline health: {health['status']}")
    
    logger.info(f"🎉 RAG Pipeline initialized successfully!")


def get_version() -> str:
    """Get pipeline version."""
    return __version__


def print_summary() -> None:
    """Print pipeline summary."""
    summary = f"""
    ╔════════════════════════════════════════════════════════╗
    ║          RAG ADMIN PIPELINE - v{__version__}                ║
    ╚════════════════════════════════════════════════════════╝
    
    📦 Modules:
       ✅ core/    - Base classes, exceptions, patterns
       ✅ cache/   - Session storage, caching
       ✅ pipeline/ - Orchestration, nodes, schemas
       ✅ utils/   - File validation, helpers
       ✅ api/     - FastAPI routes, models
    
    🔧 Main Classes:
       • PipelineOrchestrator - Main orchestration engine
       • PipelineState        - Pipeline state management
       • BaseTool             - Base class for all tools
       • CircuitBreakerManager - Fault tolerance
    
    🛠️  Key Functions:
       • get_orchestrator()   - Get orchestrator singleton
       • validate_file()      - File validation
       • measure_time()       - Performance measurement
       • format_size()        - Human-readable size
    
    📊 Pipeline Nodes:
       1. ingestion_node      - Parse files to text
       2. preprocessing_node  - Clean & normalize
       3. chunking_node       - Split into chunks
       4. embedding_node      - Generate vectors
       5. vectordb_node       - Store in database
    
    🚀 Quick Start:
       from src import get_orchestrator, validate_file
       import uuid
       
       orchestrator = get_orchestrator()
       request_id = await orchestrator.process_document(
           request_id=str(uuid.uuid4()),
           file_name="document.pdf",
           file_content=open("document.pdf", "rb").read()
       )
    
    📚 Documentation:
       https://docs.example.com/rag-pipeline
    
    """
    print(summary)


# ============================================================================
# STARTUP LOGGING
# ============================================================================

logger.info(f"✅ RAG Pipeline v{__version__} loaded successfully")
logger.debug(f"Available modules: {', '.join(__all__[:10])}...")
logger.debug(f"Total exports: {len(__all__)}")
