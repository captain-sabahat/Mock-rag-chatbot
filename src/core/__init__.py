"""
================================================================================
CORE PACKAGE - Shared Utilities and Base Classes
================================================================================

PURPOSE:
--------
Provide shared utilities, base classes, and exceptions for the RAG pipeline.

Exports:
- BaseTool: Abstract base for all tools
- CircuitBreaker: Fault tolerance mechanism  
- Exceptions: Custom exception hierarchy with severity
- Exception Severity Registry: Maps exceptions to severity levels
- Helper Functions: For CB and exception validation
- Logger setup: Centralized logging configuration

ARCHITECTURE:
--------------
All modules
    ↓
src/core/ ← This package (shared utilities)
    ↓
Used by: api/, pipeline/, tools/, cache/

NO BUSINESS LOGIC: This layer only provides utilities and abstractions.

KEY COMPONENTS:
--------------
1. BaseTool - Base class for all tools with optional monitoring hooks
2. CircuitBreakerManager - Fault tolerance with orchestrator integration
3. ExceptionSeverity - INFO, WARNING, CRITICAL levels
4. Exception Classes - ValidationError, ProcessingError, etc.
5. Severity Registry - Maps exception types to severity
6. Helper Functions:
   - get_exception_severity()
   - should_trigger_circuit_breaker()
   - validate_exception_severity()
   - is_exception_recoverable()
   - check_condition_d()

USAGE EXAMPLE:
--------------
from src.core import (
    BaseTool,
    CircuitBreakerManager,
    ExceptionSeverity,
    should_trigger_circuit_breaker
)

class MyTool(BaseTool):
    pass

================================================================================
"""

from .base_tool import BaseTool, ToolConfig
from .circuit_breaker import (
    CircuitBreakerManager,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreaker,
)
from .exceptions import (
    RAGPipelineException,
    ValidationError,
    ProcessingError,
    ChunkingError,
    EmbeddingError,
    VectorDBError,
    ConfigurationError,
    CircuitBreakerError,
    ExceptionSeverity,
    # Helper functions
    get_exception_severity,
    should_trigger_circuit_breaker,
    validate_exception_severity,
    is_exception_recoverable,
    log_exception_severity,
    register_exception_severity,
    get_severity_summary,
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolConfig",
    
    # Circuit breaker
    "CircuitBreakerManager",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitBreaker",
    
    # Exception severity
    "ExceptionSeverity",
    
    # Exceptions
    "RAGPipelineException",
    "ValidationError",
    "ProcessingError",
    "ChunkingError",
    "EmbeddingError",
    "VectorDBError",
    "ConfigurationError",
    "CircuitBreakerError",
    
    # Helper functions
    "get_exception_severity",
    "should_trigger_circuit_breaker",
    "validate_exception_severity",
    "is_exception_recoverable",
    "log_exception_severity",
    "register_exception_severity",
    "get_severity_summary",
]
