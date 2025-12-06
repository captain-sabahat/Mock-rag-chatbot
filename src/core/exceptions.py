"""
================================================================================
EXCEPTIONS - Custom Exception Hierarchy
================================================================================

PURPOSE:
--------
Define custom exceptions for the RAG pipeline.

Hierarchy:
  RAGPipelineException (base)
    ├── ValidationError
    ├── ProcessingError
    │   ├── ChunkingError
    │   ├── EmbeddingError
    │   └── VectorDBError
    └── ConfigurationError

Benefits:
  ✅ Easy error handling (catch specific exceptions)
  ✅ Clear error messages
  ✅ Structured error information
  ✅ Better debugging

USAGE:
------
    try:
        result = await chunker.execute(text=doc)
    except ChunkingError as e:
        logger.error(f"Chunking failed: {e.message}")
    except ProcessingError as e:
        logger.error(f"Processing error: {e.details}")
    except RAGPipelineException as e:
        logger.error(f"Unknown RAG error: {e}")

================================================================================
"""

from typing import Dict, Any, Optional


class RAGPipelineException(Exception):
    """
    Base exception for all RAG pipeline errors.
    
    Attributes:
        error_code: Machine-readable error identifier
        message: Human-readable error message
        details: Additional error context (dict)
    """

    def __init__(
        self,
        message: str,
        error_code: str = "rag_pipeline_error",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            error_code: Error identifier
            details: Additional context
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dict (for JSON responses)."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }

    def __str__(self) -> str:
        """String representation."""
        if self.details:
            return f"{self.error_code}: {self.message} {self.details}"
        return f"{self.error_code}: {self.message}"


class ValidationError(RAGPipelineException):
    """
    Raised when input validation fails.
    
    Examples:
      - Missing required parameters
      - Invalid file format
      - Invalid configuration
      - Type mismatch
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize validation error."""
        super().__init__(
            message=message,
            error_code="validation_error",
            details=details
        )


class ProcessingError(RAGPipelineException):
    """
    Base class for pipeline processing errors.
    
    Raised when a step in the pipeline fails.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize processing error."""
        super().__init__(
            message=message,
            error_code="processing_error",
            details=details
        )


class ChunkingError(ProcessingError):
    """
    Raised when document chunking fails.
    
    Examples:
      - Text splitting algorithm error
      - Invalid chunk parameters
      - Memory error with large document
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize chunking error."""
        super().__init__(
            message=message,
            details=details
        )
        self.error_code = "chunking_error"


class EmbeddingError(ProcessingError):
    """
    Raised when embedding generation fails.
    
    Examples:
      - API call failure
      - Model loading error
      - Invalid embedding dimension
      - Rate limit exceeded
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize embedding error."""
        super().__init__(
            message=message,
            details=details
        )
        self.error_code = "embedding_error"


class VectorDBError(ProcessingError):
    """
    Raised when vector database operations fail.
    
    Examples:
      - Connection error
      - Index not found
      - Invalid vector format
      - Storage failure
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize vector DB error."""
        super().__init__(
            message=message,
            details=details
        )
        self.error_code = "vectordb_error"


class ConfigurationError(RAGPipelineException):
    """
    Raised when configuration is invalid.
    
    Examples:
      - Missing required config key
      - Invalid backend selection
      - Type mismatch in config
      - Incompatible settings
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize configuration error."""
        super().__init__(
            message=message,
            error_code="configuration_error",
            details=details
        )


# Convenience functions for common errors

def raise_validation_error(message: str, **details) -> None:
    """Raise a validation error."""
    raise ValidationError(message, details=details)


def raise_chunking_error(message: str, **details) -> None:
    """Raise a chunking error."""
    raise ChunkingError(message, details=details)


def raise_embedding_error(message: str, **details) -> None:
    """Raise an embedding error."""
    raise EmbeddingError(message, details=details)


def raise_vectordb_error(message: str, **details) -> None:
    """Raise a vector DB error."""
    raise VectorDBError(message, details=details)


def raise_config_error(message: str, **details) -> None:
    """Raise a configuration error."""
    raise ConfigurationError(message, details=details)
