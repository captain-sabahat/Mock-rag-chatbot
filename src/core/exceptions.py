"""
================================================================================
CORE EXCEPTIONS - Exception Hierarchy & Severity Registry
================================================================================

PURPOSE:
--------
Define exception hierarchy and severity levels for circuit breaker logic.

Responsibilities:
- Exception definitions with severity classification
- Severity registry mapping exception types to levels
- Helper functions for exception validation
- Integration with circuit breaker conditions (A-D)

Exception Hierarchy:
BaseError (inherited from Exception)
├── ValidationError (input/output validation failures)
├── ProcessingError (node execution failures)
├── CircuitBreakerError (circuit breaker triggered)
├── EmbeddingError (embedding generation failures)
└── VectorDBError (vector database failures)

Severity Levels:
- INFO: Non-critical warnings (can continue)
- WARNING: May affect quality but recoverable
- CRITICAL: Breaks output quality, must stop (triggers CB)

Circuit Breaker Integration (Condition A):
If exception_severity == "CRITICAL" → OPEN

================================================================================
"""

from enum import Enum
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ================================================================================
# EXCEPTION SEVERITY LEVELS
# ================================================================================

class ExceptionSeverity(str, Enum):
    """Exception severity levels for circuit breaker decisions."""
    
    INFO = "INFO"          # Informational, non-critical
    WARNING = "WARNING"    # Warning, may affect quality
    CRITICAL = "CRITICAL" # Critical error, must stop (triggers circuit breaker)

# ================================================================================
# EXCEPTION DEFINITIONS
# ================================================================================

class BaseError(Exception):
    """Base error class with message and details."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        self.severity = ExceptionSeverity.CRITICAL  # Default
        super().__init__(self.message)

class ValidationError(BaseError):
    """Raised when input/output validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.severity = ExceptionSeverity.CRITICAL  # Always critical

class ProcessingError(BaseError):
    """Raised when node processing fails."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.severity = ExceptionSeverity.CRITICAL  # Always critical

class CircuitBreakerError(BaseError):
    """Raised when circuit breaker is triggered."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.severity = ExceptionSeverity.CRITICAL  # Always critical

class EmbeddingError(BaseError):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.severity = ExceptionSeverity.CRITICAL  # Always critical

class VectorDBError(BaseError):
    """Raised when vector database operation fails."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.severity = ExceptionSeverity.CRITICAL  # Always critical

class ChunkingError(BaseError):
    """Raised when chunking fails."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.severity = ExceptionSeverity.CRITICAL

class ConfigurationError(BaseError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.severity = ExceptionSeverity.CRITICAL

class RAGPipelineException(BaseError):
    """Generic RAG pipeline exception."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.severity = ExceptionSeverity.CRITICAL

# ================================================================================
# EXCEPTION SEVERITY REGISTRY
# ================================================================================

# Maps exception types to severity levels
# Used by circuit breaker to determine if exception should trigger break

EXCEPTION_SEVERITY_REGISTRY: Dict[str, ExceptionSeverity] = {
    # Input/Output validation errors → CRITICAL
    "MissingInputError": ExceptionSeverity.CRITICAL,
    "TypeValidationError": ExceptionSeverity.CRITICAL,
    "EmptyOutputError": ExceptionSeverity.CRITICAL,
    "OutputValidationError": ExceptionSeverity.CRITICAL,
    "ValueError": ExceptionSeverity.CRITICAL,
    "InvalidInputError": ExceptionSeverity.CRITICAL,
    
    # Data format errors → CRITICAL
    "SchemaValidationError": ExceptionSeverity.CRITICAL,
    "JSONDecodeError": ExceptionSeverity.CRITICAL,
    "KeyError": ExceptionSeverity.CRITICAL,
    "DataMismatchError": ExceptionSeverity.CRITICAL,
    
    # Timeout/Async errors → CRITICAL
    "TimeoutError": ExceptionSeverity.CRITICAL,
    "asyncio.TimeoutError": ExceptionSeverity.CRITICAL,
    "ConnectionError": ExceptionSeverity.CRITICAL,
    "TimeoutException": ExceptionSeverity.CRITICAL,
    
    # Model/Tool errors → CRITICAL
    "ModelLoadError": ExceptionSeverity.CRITICAL,
    "EmbeddingError": ExceptionSeverity.CRITICAL,
    "VectorDBError": ExceptionSeverity.CRITICAL,
    "ParseError": ExceptionSeverity.CRITICAL,
    "ProcessingError": ExceptionSeverity.CRITICAL,
    "ChunkingError": ExceptionSeverity.CRITICAL,
    
    # Validation errors → CRITICAL
    "ValidationError": ExceptionSeverity.CRITICAL,
    
    # Warning: Recoverable issues → WARNING
    "DeprecationWarning": ExceptionSeverity.WARNING,
    "ResourceWarning": ExceptionSeverity.WARNING,
    "LowQualityWarning": ExceptionSeverity.WARNING,
    
    # Info: Not critical → INFO
    "UserWarning": ExceptionSeverity.INFO,
    "LoggingWarning": ExceptionSeverity.INFO,
    
    # Circuit breaker → CRITICAL
    "CircuitBreakerError": ExceptionSeverity.CRITICAL,
    
    # Configuration → CRITICAL
    "ConfigurationError": ExceptionSeverity.CRITICAL,
    "RAGPipelineException": ExceptionSeverity.CRITICAL,
}

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def get_exception_severity(exception_type: str) -> ExceptionSeverity:
    """
    Get severity level for an exception type.
    
    Default to CRITICAL if unknown (fail-safe approach).
    Unknown exceptions are treated as critical to avoid data loss.
    
    Args:
        exception_type: Name of the exception type
    
    Returns:
        ExceptionSeverity: Severity level
    
    Usage:
        severity = get_exception_severity("ValueError")
        # Returns: ExceptionSeverity.CRITICAL
    """
    return EXCEPTION_SEVERITY_REGISTRY.get(
        exception_type,
        ExceptionSeverity.CRITICAL  # Fail-safe: unknown exceptions are CRITICAL
    )

def validate_exception_severity(
    exception_type: str,
    expected_severity: ExceptionSeverity
) -> bool:
    """
    Validate that exception has expected severity.
    
    Useful for testing and assertion.
    
    Args:
        exception_type: Name of exception
        expected_severity: Expected severity level
    
    Returns:
        bool: True if severity matches
    
    Usage:
        assert validate_exception_severity("ValueError", ExceptionSeverity.CRITICAL)
    """
    actual_severity = get_exception_severity(exception_type)
    return actual_severity == expected_severity

def should_trigger_circuit_breaker(
    exception_type: str,
    severity: Optional[ExceptionSeverity] = None
) -> bool:
    """
    Determine if exception should trigger circuit breaker.
    
    Rule: CRITICAL exceptions ALWAYS trigger circuit breaker
    
    This implements Condition A of the circuit breaker logic:
    A: CRITICAL Exception → OPEN
    
    Args:
        exception_type: Name of the exception
        severity: Optional pre-computed severity
    
    Returns:
        bool: True if circuit breaker should trigger
    
    Usage:
        if should_trigger_circuit_breaker("EmbeddingError"):
            # Trigger circuit breaker
            pass
    """
    if severity is None:
        severity = get_exception_severity(exception_type)
    
    return severity == ExceptionSeverity.CRITICAL

def is_exception_recoverable(severity: ExceptionSeverity) -> bool:
    """
    Check if exception is recoverable.
    
    Args:
        severity: Exception severity level
    
    Returns:
        bool: True if exception is recoverable
    
    Usage:
        if not is_exception_recoverable(exception.severity):
            # Need to fail the pipeline
            pass
    """
    return severity in [ExceptionSeverity.INFO, ExceptionSeverity.WARNING]

def log_exception_severity(
    exception_type: str,
    message: str,
    logger_instance=None
) -> ExceptionSeverity:
    """
    Get severity and log it with emoji indicator.
    
    Args:
        exception_type: Name of exception
        message: Error message
        logger_instance: Logger instance (optional)
    
    Returns:
        ExceptionSeverity: Computed severity
    
    Usage:
        severity = log_exception_severity(
            "ValueError",
            "Invalid input format",
            logger
        )
    """
    severity = get_exception_severity(exception_type)
    
    if logger_instance:
        if severity == ExceptionSeverity.CRITICAL:
            logger_instance.error(f"🔴 CRITICAL [{exception_type}]: {message}")
        elif severity == ExceptionSeverity.WARNING:
            logger_instance.warning(f"🟡 WARNING [{exception_type}]: {message}")
        else:
            logger_instance.info(f"🟢 INFO [{exception_type}]: {message}")
    
    return severity

def register_exception_severity(
    exception_type: str,
    severity: ExceptionSeverity
) -> None:
    """
    Register custom exception severity (extensible registry).
    
    Args:
        exception_type: Exception type name
        severity: Severity level
    
    Usage:
        register_exception_severity("CustomError", ExceptionSeverity.WARNING)
    """
    EXCEPTION_SEVERITY_REGISTRY[exception_type] = severity
    logger.info(f"Registered {exception_type} as {severity.value}")

def get_severity_summary() -> Dict[str, list]:
    """
    Get summary of all registered exception severities.
    
    Returns:
        Dict mapping severity to list of exception types
    """
    summary = {
        "CRITICAL": [],
        "WARNING": [],
        "INFO": []
    }
    
    for exc_type, severity in EXCEPTION_SEVERITY_REGISTRY.items():
        summary[severity.value].append(exc_type)
    
    return summary
