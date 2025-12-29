# MERGED: 3 sections with separation comments
#│   │   ├── SECTION 1: Base exceptions
#│   │   ├── SECTION 2: Service exceptions
#│   │   └── SECTION 3: API exceptions
"""
================================================================================
FILE: src/exceptions.py
================================================================================

PURPOSE:
    Custom exception hierarchy for the entire RAG backend. Defines all exception
    types used throughout the application, enabling structured error handling,
    recovery strategies, and logging.

WORKFLOW:
    1. Define base exception class (RAGPipelineException)
    2. Define exception categories:
       - RecoverableException: Error can be retried (transient failure)
       - FatalException: Error cannot be recovered (fatal failure)
    3. Define specific exception types for each service:
       - Redis errors (cache failures)
       - Vector DB errors (search failures)
       - LLM/SLM errors (model inference failures)
       - Document errors (parsing failures)
       - API errors (validation failures)

IMPORTS:
    - None (only Python builtins)

INPUTS:
    - Exception message (str)
    - Error code (str) for categorization
    - Optional context dict (for logging)

OUTPUTS:
    - Exception instances (raised by handlers/pipelines)

KEY FACTS:
    - NO imports from src modules (prevents circular dependencies)
    - All exceptions inherit from RAGPipelineException
    - Each exception has error_code for categorization
    - Circuit breaker uses exception type to decide recovery strategy
    - Recoverable exceptions trigger retries; fatal exceptions fail fast

EXCEPTION CATEGORIES:
    - RECOVERABLE (transient, should retry):
        * RedisError: Cache timeout, connection reset
        * VectorDBError: Search timeout, connection issues
        * LLMError: API timeout, rate limit, temporary unavailable
        * DocumentProcessingError: Parsing failure (may succeed on retry)
        * TimeoutError: Operation exceeded time limit
    
    - FATAL (non-recoverable, fail fast):
        * ModelInitializationError: Model failed to load
        * ConfigurationError: Invalid configuration
        * ValidationError: Invalid input data (won't fix on retry)

FUTURE SCOPE (Phase 2+):
    - Add context traceback (for detailed error diagnostics)
    - Add error code documentation (map codes to solutions)
    - Add sentry integration (automatic error reporting)
    - Add error metrics (track by error type)
    - Add user-friendly error messages (vs technical messages)
    - Add error recovery suggestions (what to do next)
    - Add error context serialization (for logging)
    - Add deprecation warnings (for API changes)

TESTING ENVIRONMENT:
    - Mock exceptions in tests (raise specific exception types)
    - Test circuit breaker behavior with RecoverableException
    - Test fast-fail with FatalException
    - Verify error codes are correctly assigned
    - Test exception propagation through layers
"""

# ================================================================================
# IMPORTS
# ================================================================================

from typing import Optional, Dict, Any

# ================================================================================
# SECTION 1: BASE EXCEPTIONS
# ================================================================================

class RAGPipelineException(Exception):
    """
    Root exception for all RAG pipeline errors.
    
    All custom exceptions inherit from this for consistent handling.
    
    Attributes:
        message (str): Human-readable error message
        error_code (str): Machine-readable error code for categorization
        context (dict): Additional context (optional)
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dict for JSON response"""
        return {
            "error": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }


class RecoverableException(RAGPipelineException):
    """
    Exception that allows retry logic.
    
    Used for transient failures (timeouts, connection errors, etc.)
    Circuit breaker will retry when this is raised.
    
    FUTURE EXTENSION (Phase 2):
        - Add retry_count parameter
        - Add backoff_strategy (exponential, linear)
        - Add max_retries (configurable)
    """
    pass


class FatalException(RAGPipelineException):
    """
    Exception that cannot be recovered.
    
    Used for permanent failures (config errors, model load failures, etc.)
    Circuit breaker will FAIL FAST (not retry) when this is raised.
    
    FUTURE EXTENSION (Phase 2):
        - Add fallback service suggestion
        - Add recovery action recommendation
    """
    pass

# ================================================================================
# SECTION 2: SERVICE EXCEPTIONS
# ================================================================================

class RedisError(RecoverableException):
    """Redis cache operation failure (can retry)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="REDIS_ERROR", context=context)


class VectorDBError(RecoverableException):
    """Vector database operation failure (can retry)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="VECTOR_DB_ERROR", context=context)


class LLMError(RecoverableException):
    """LLM inference failure (can retry)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="LLM_ERROR", context=context)


class SLMError(RecoverableException):
    """SLM inference failure (can retry)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="SLM_ERROR", context=context)


class DocumentProcessingError(RecoverableException):
    """Document parsing/summarization failure (can retry)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="DOCUMENT_ERROR", context=context)


class EmbeddingError(RecoverableException):
    """Embedding generation failure (can retry)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="EMBEDDING_ERROR", context=context)


class TimeoutError(RecoverableException):
    """Operation exceeded timeout (can retry)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="TIMEOUT_ERROR", context=context)


class ServiceUnavailableError(RecoverableException):
    """Service temporarily unavailable (can retry)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="SERVICE_UNAVAILABLE", context=context)


class CircuitBreakerOpenError(RecoverableException):
    """Circuit breaker is open (fail fast, retry later)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="CIRCUIT_BREAKER_OPEN", context=context)

# ================================================================================
# SECTION 3: API & VALIDATION EXCEPTIONS
# ================================================================================

class ValidationError(FatalException):
    """Request validation failed (won't fix on retry)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="VALIDATION_ERROR", context=context)


class ModelInitializationError(FatalException):
    """Model failed to initialize (fatal)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="MODEL_INIT_ERROR", context=context)


class ConfigurationError(FatalException):
    """Invalid configuration (fatal)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="CONFIG_ERROR", context=context)


class SLMTimeoutError(RecoverableException):
    """SLM summarization timeout (can retry or skip)"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="SLM_TIMEOUT", context=context)


class DocumentTooLargeError(ValidationError):
    """Document exceeds size limit"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="DOCUMENT_TOO_LARGE", context=context)


class UnsupportedFileTypeError(ValidationError):
    """File type not supported"""
    
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="UNSUPPORTED_FILE_TYPE", context=context)

class ServiceInitializationError(FatalException):
    """Raised when a service/provider fails to initialize (fatal)."""

    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message, error_code="SERVICE_INIT_ERROR", context=context)
