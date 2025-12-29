"""
================================================================================
FILE: src/config/constants.py
================================================================================

PURPOSE:
    Application-wide constants. Immutable values used throughout codebase.
    Enables consistency and prevents magic numbers.

WORKFLOW:
    1. Define all constants (no computation, just values)
    2. Organize by category
    3. Use throughout app: from src.config import CONSTANTS
    4. Never modify constants at runtime

IMPORTS:
    - None (only builtins)

INPUTS:
    - None (no runtime input)

OUTPUTS:
    - CONSTANTS dict with all application constants

CONSTANT CATEGORIES:
    1. API Configuration
       - API_VERSION: "v1"
       - API_PREFIX: "/api/v1"
       - API_TITLE: "User Chatbot RAG Backend"
    
    2. Latency Targets
       - LATENCY_LOGIC_1_MS: <10ms (cache hit)
       - LATENCY_LOGIC_2_MS: <2s (pure RAG)
       - LATENCY_LOGIC_3_MS: <3s (cache + doc)
       - LATENCY_LOGIC_4_MS: <4s (RAG + doc)
       - MAX_REQUEST_TIMEOUT_MS: 30s
    
    3. Cache Keys
       - CACHE_KEY_QUERY: "user:{user_id}:query:{query_hash}"
       - CACHE_KEY_SESSION: "session:{user_id}:{session_id}"
       - CACHE_KEY_SUMMARY: "summary:{user_id}:{doc_hash}"
    
    4. Document Limits
       - DOC_MAX_SIZE_BYTES: 50MB
       - DOC_MIN_SIZE_BYTES: 1 byte
       - DOC_CHUNK_SIZE_DEFAULT: 1000 characters
    
    5. Text Limits
       - PROMPT_MAX_LENGTH: 5000
       - PROMPT_MIN_LENGTH: 1
       - ANSWER_MAX_LENGTH: 10000
    
    6. Batch Processing
       - EMBEDDINGS_BATCH_SIZE_DEFAULT: 32
       - EMBEDDINGS_MAX_BATCH_WAIT_MS: 10ms
    
    7. Retry Configuration
       - RETRY_MAX_ATTEMPTS: 3
       - RETRY_BACKOFF_FACTOR: 2.0 (exponential)
    
    8. Circuit Breaker
       - CIRCUIT_BREAKER_FAILURE_THRESHOLD: 3 failures
       - CIRCUIT_BREAKER_RECOVERY_TIMEOUT: 60s

KEY FACTS:
    - All constants are immutable (enforced by Python)
    - No computation, just literal values
    - No imports from other src modules (prevent circular deps)
    - Used for configuration, not business logic
    - Should be inspected by logging/monitoring

FUTURE SCOPE (Phase 2+):
    - Add feature flag constants
    - Add A/B test variant constants
    - Add monitoring threshold constants
    - Add metric name constants
    - Add error code constants

TESTING ENVIRONMENT:
    - Reference constants in tests
    - Verify values haven't changed
    - Use in test assertions

PRODUCTION DEPLOYMENT:
    - Constants never change at runtime
    - Safe to use across all environments
    - Good source for documentation
"""

# ================================================================================
# API CONFIGURATION
# ================================================================================

API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
API_TITLE = "User Chatbot RAG Backend"
API_DESCRIPTION = "Production-grade RAG pipeline for user chatbot"

# ================================================================================
# LATENCY TARGETS (milliseconds)
# ================================================================================

# LOGIC 1: Redis cache hit only (ultra-fast)
LATENCY_LOGIC_1_TARGET_MS = 10
LATENCY_LOGIC_1_P99_MS = 50

# LOGIC 2: Pure RAG (embeddings + vector DB + LLM)
LATENCY_LOGIC_2_TARGET_MS = 2000
LATENCY_LOGIC_2_P99_MS = 3000

# LOGIC 3: Redis cache + Document summary
LATENCY_LOGIC_3_TARGET_MS = 3000
LATENCY_LOGIC_3_P99_MS = 4000

# LOGIC 4: RAG + Document summary
LATENCY_LOGIC_4_TARGET_MS = 4000
LATENCY_LOGIC_4_P99_MS = 5000

# Global timeout for any request
MAX_REQUEST_TIMEOUT_MS = 30000  # 30 seconds
MAX_REQUEST_TIMEOUT_SECONDS = MAX_REQUEST_TIMEOUT_MS / 1000

# ================================================================================
# CACHE KEY TEMPLATES
# ================================================================================

CACHE_KEY_QUERY = "user:{user_id}:query:{query_hash}"
CACHE_KEY_SESSION = "session:{user_id}:{session_id}"
CACHE_KEY_SUMMARY = "summary:{user_id}:{doc_hash}"

# Cache key prefixes for easy filtering
CACHE_KEY_PREFIX_USER = "user:"
CACHE_KEY_PREFIX_SESSION = "session:"
CACHE_KEY_PREFIX_SUMMARY = "summary:"

# ================================================================================
# DOCUMENT LIMITS
# ================================================================================

DOC_MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
DOC_MIN_SIZE_BYTES = 1  # 1 byte
DOC_CHUNK_SIZE_DEFAULT = 1000  # characters

# Supported document formats
DOC_SUPPORTED_FORMATS = ["pdf", "docx", "txt"]

# ================================================================================
# TEXT LIMITS
# ================================================================================

PROMPT_MAX_LENGTH = 5000
PROMPT_MIN_LENGTH = 1
ANSWER_MAX_LENGTH = 10000
SUMMARY_MAX_LENGTH = 1000

# ================================================================================
# BATCH PROCESSING
# ================================================================================

EMBEDDINGS_BATCH_SIZE_DEFAULT = 32
EMBEDDINGS_MAX_BATCH_WAIT_MS = 10
EMBEDDINGS_MAX_BATCH_WAIT_SECONDS = EMBEDDINGS_MAX_BATCH_WAIT_MS / 1000

# ================================================================================
# RETRY CONFIGURATION
# ================================================================================

RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 2.0  # Exponential backoff
RETRY_MAX_BACKOFF_SECONDS = 60.0

# ================================================================================
# CIRCUIT BREAKER
# ================================================================================

CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3  # Open after 3 failures
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS = 60  # Retry after 60s
CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS = 1  # Try once in half-open state

# ================================================================================
# ERROR CODES
# ================================================================================

ERROR_REDIS_CONNECTION = "REDIS_CONNECTION_ERROR"
ERROR_VECTOR_DB_CONNECTION = "VECTOR_DB_CONNECTION_ERROR"
ERROR_LLM_INFERENCE = "LLM_INFERENCE_ERROR"
ERROR_SLM_INFERENCE = "SLM_INFERENCE_ERROR"
ERROR_EMBEDDING_GENERATION = "EMBEDDING_GENERATION_ERROR"
ERROR_DOCUMENT_PARSING = "DOCUMENT_PARSING_ERROR"
ERROR_REQUEST_VALIDATION = "REQUEST_VALIDATION_ERROR"
ERROR_REQUEST_TIMEOUT = "REQUEST_TIMEOUT_ERROR"
ERROR_SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE_ERROR"

# ================================================================================
# METRIC NAMES (for Prometheus)
# ================================================================================

METRIC_REQUEST_COUNT = "rag_pipeline_requests_total"
METRIC_REQUEST_LATENCY = "rag_pipeline_latency_ms"
METRIC_ERROR_COUNT = "rag_pipeline_errors_total"
METRIC_CACHE_HIT_RATE = "rag_pipeline_cache_hit_rate"
METRIC_REDIS_LATENCY = "redis_latency_ms"
METRIC_VECTOR_DB_LATENCY = "vector_db_latency_ms"
METRIC_LLM_LATENCY = "llm_latency_ms"
METRIC_SLM_LATENCY = "slm_latency_ms"

# ================================================================================
# LOG LEVELS
# ================================================================================

LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"

# ================================================================================
# DEVICE TYPES
# ================================================================================

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"  # Apple Silicon
DEVICE_NPU = "npu"  # Neural Processing Unit
DEVICE_AUTO = "auto"

SUPPORTED_DEVICES = [DEVICE_CPU, DEVICE_CUDA, DEVICE_MPS, DEVICE_NPU, DEVICE_AUTO]

# ================================================================================
# QUANTIZATION TYPES
# ================================================================================

QUANTIZATION_NONE = "none"
QUANTIZATION_INT8 = "int8"  # 50% memory reduction
QUANTIZATION_INT4 = "int4"  # 75% memory reduction

SUPPORTED_QUANTIZATION = [QUANTIZATION_NONE, QUANTIZATION_INT8, QUANTIZATION_INT4]

# ================================================================================
# CONSOLIDATE CONSTANTS DICT
# ================================================================================

CONSTANTS = {
    # API
    "api_version": API_VERSION,
    "api_prefix": API_PREFIX,
    "api_title": API_TITLE,
    
    # Latency targets
    "latency_logic_1_target_ms": LATENCY_LOGIC_1_TARGET_MS,
    "latency_logic_2_target_ms": LATENCY_LOGIC_2_TARGET_MS,
    "latency_logic_3_target_ms": LATENCY_LOGIC_3_TARGET_MS,
    "latency_logic_4_target_ms": LATENCY_LOGIC_4_TARGET_MS,
    "max_request_timeout_ms": MAX_REQUEST_TIMEOUT_MS,
    
    # Cache
    "cache_key_query": CACHE_KEY_QUERY,
    "cache_key_session": CACHE_KEY_SESSION,
    "cache_key_summary": CACHE_KEY_SUMMARY,
    
    # Document
    "doc_max_size_bytes": DOC_MAX_SIZE_BYTES,
    "doc_supported_formats": DOC_SUPPORTED_FORMATS,
    
    # Text
    "prompt_max_length": PROMPT_MAX_LENGTH,
    "answer_max_length": ANSWER_MAX_LENGTH,
    
    # Retry
    "retry_max_attempts": RETRY_MAX_ATTEMPTS,
    "retry_backoff_factor": RETRY_BACKOFF_FACTOR,
    
    # Circuit breaker
    "circuit_breaker_failure_threshold": CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    "circuit_breaker_recovery_timeout": CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
    
    # Devices
    "supported_devices": SUPPORTED_DEVICES,
    
    # Quantization
    "supported_quantization": SUPPORTED_QUANTIZATION,
}
