"""
================================================================================
RAG ADMIN PIPELINE - Root Package Initialization
================================================================================

PURPOSE
-------
Expose the main public API of the RAG Admin Pipeline from a single root package
`src`, so application code can do:

    from src import get_orchestrator, validate_file, BaseTool

instead of importing from many internal modules.

LAYERS
------
- src.core      : BaseTool, exceptions, circuit breaker, log buffer
- src.pipeline  : Orchestrator, pipeline state/schemas, nodes
- src.api       : FastAPI routers and API models
- src.utils     : Generic utilities (validation, time, formatting, etc.)
- src.cache     : Session store abstraction (if you use it)

KEY OBJECTS
-----------
- Orchestrator:
    get_orchestrator(), PipelineOrchestrator

- Pipeline models:
    PipelineState, NodeStatus, PipelineStatus, CircuitBreakerState

- Core:
    BaseTool, ToolConfig
    RAGPipelineException + specific error types
    CircuitBreakerManager, CircuitState

- Monitoring API:
    api_router (FastAPI router with /api/monitor/* and other endpoints)

QUICK START
-----------
from src import get_orchestrator, validate_file
import uuid

# Validate file bytes
is_valid, error = validate_file("document.pdf", file_bytes)
if not is_valid:
    raise ValueError(error)

# Run pipeline
orchestrator = get_orchestrator()
request_id = str(uuid.uuid4())
result_id = orchestrator.process_document_sync(
    request_id=request_id,
    file_name="document.pdf",
    file_content=file_bytes,
)

print("Request ID:", result_id)

# In a different process / later:
# Use /api/monitor/status/{request_id} to read monitoring JSON files.

================================================================================
"""

from __future__ import annotations

import logging
from typing import Any

# -----------------------------------------------------------------------------
# Basic package metadata and logging
# -----------------------------------------------------------------------------

__version__ = "1.0.0"
__title__ = "RAG Admin Pipeline"
__author__ = "RAG Pipeline Team"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("🚀 Initializing %s v%s", __title__, __version__)

# -----------------------------------------------------------------------------
# CORE IMPORTS - Base classes, exceptions, circuit breaker, log buffer
# -----------------------------------------------------------------------------

from src.core import (
    # Base tool abstraction
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
    # Exception severity helpers (used by orchestrator & CB)
    get_exception_severity,
    should_trigger_circuit_breaker,
    validate_exception_severity,
    is_exception_recoverable,
    register_exception_severity,
    # Circuit breaker
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
)

# Optional: if you have a log buffer class in core/log_buffer.py
try:  # keep optional so src can import without api/log deps
    from src.core.log_buffer import LogBuffer
except Exception:  # pragma: no cover - optional
    LogBuffer = None  # type: ignore


# -----------------------------------------------------------------------------
# PIPELINE IMPORTS - Orchestrator, state/schemas, nodes package
# -----------------------------------------------------------------------------

from src.pipeline.schemas import (
    PipelineState,
    NodeStatus,
    PipelineStatus,
    CircuitBreakerState,  # overall CB snapshot used in monitoring
)

from src.pipeline import (
    PipelineOrchestrator,
    get_orchestrator,
    reset_orchestrator,
    nodes,  # src.pipeline.nodes package (ingestion_node, etc.)
)


# -----------------------------------------------------------------------------
# UTILS IMPORTS - Validation, helpers, formatting, retry, etc.
# -----------------------------------------------------------------------------

from src.utils import (
    # File validation
    validate_file,
    validate_file_type,
    validate_file_size,
    get_file_type,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    # Timing / metrics
    measure_time,
    format_size,
    format_duration,
    safe_json_dumps,
    retry_on_exception,
    # Misc
    sanitize_filename,
)


# -----------------------------------------------------------------------------
# API IMPORTS - Routers and models (optional at import time)
# -----------------------------------------------------------------------------

try:
    from src.api import (
        router as api_router,
        # If you still expose these Pydantic models:
        UploadRequest,
        UploadResponse,
        QueryRequest,
        QueryResponse,
        StatusResponse,
        ErrorResponse,
    )
except Exception as e:  # pragma: no cover - API optional for non‑web usage
    logger.warning("⚠️ API imports failed (optional): %s", e)
    api_router = None  # type: ignore
    UploadRequest = UploadResponse = QueryRequest = QueryResponse = None  # type: ignore
    StatusResponse = ErrorResponse = None  # type: ignore


# -----------------------------------------------------------------------------
# PUBLIC API - What `from src import *` exposes
# -----------------------------------------------------------------------------

__all__ = [
    # Metadata
    "__version__",
    "__title__",
    "__author__",
    # Core tool + config
    "BaseTool",
    "ToolConfig",
    # Exceptions
    "RAGPipelineException",
    "ValidationError",
    "ProcessingError",
    "ChunkingError",
    "EmbeddingError",
    "VectorDBError",
    "ConfigurationError",
    # Exception helpers
    "get_exception_severity",
    "should_trigger_circuit_breaker",
    "validate_exception_severity",
    "is_exception_recoverable",
    "register_exception_severity",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerManager",
    "CircuitState",
    # Optional log buffer
    "LogBuffer",
    # Pipeline orchestrator and state
    "PipelineOrchestrator",
    "get_orchestrator",
    "reset_orchestrator",
    "PipelineState",
    "NodeStatus",
    "PipelineStatus",
    "CircuitBreakerState",
    # Nodes package (src.pipeline.nodes)
    "nodes",
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
    # Helpers below
    "initialize_pipeline",
    "get_version",
    "print_summary",
]


# -----------------------------------------------------------------------------
# INITIALIZATION HELPERS
# -----------------------------------------------------------------------------

def initialize_pipeline(
    log_level: str = "INFO",
) -> None:
    """
    Initialize the RAG pipeline.

    - Sets global logging level.
    - Creates the orchestrator singleton (which in turn wires circuit breaker,
      monitoring writer, and node graph).
    - Can be called once at application startup.
    """
    level = log_level.upper()
    logging.getLogger().setLevel(getattr(logging, level, logging.INFO))
    logger.info("📋 Logging level set to %s", level)

    # Touch orchestrator so any lazy initialization runs
    _ = get_orchestrator()
    logger.info("🚀 Pipeline orchestrator initialized")


def get_version() -> str:
    """Return the package version string."""
    return __version__


def print_summary() -> None:
    """Print a high‑level summary of the RAG pipeline components."""
    summary = f"""
╔════════════════════════════════════════════════════════╗
║ {__title__} - v{__version__} ║
╚════════════════════════════════════════════════════════╝

📦 Modules:
  • core/      - BaseTool, exceptions, circuit breaker, log buffer
  • pipeline/  - Orchestrator, nodes, schemas, monitoring integration
  • utils/     - File validation, helpers, timing, retries
  • api/       - FastAPI routes & monitoring endpoints

🧠 Orchestrator:
  • PipelineOrchestrator (get via get_orchestrator())
  • Writes monitoring JSON:
      data/monitoring/nodes/{{request_id}}/{{node}}_node.json
      data/monitoring/nodes/{{request_id}}/pipeline_status.json

🧩 Core:
  • BaseTool, ToolConfig
  • CircuitBreakerManager, CircuitState
  • Exception severity helpers (should_trigger_circuit_breaker, ...)

📊 Monitoring API:
  • /api/monitor/status/{{request_id}}
  • /api/monitor/metrics
  • /api/monitor/health
  • /api/monitor/circuit-breaker
"""
    print(summary)
