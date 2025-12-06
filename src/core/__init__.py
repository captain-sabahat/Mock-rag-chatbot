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
  - Exceptions: Custom exception hierarchy
  - Logger setup: Centralized logging configuration

ARCHITECTURE:
--------------
     All modules
           ↓
     src/core/  ← This package (shared utilities)
           ↓
     Used by: api/, pipeline/, tools/, cache/

NO BUSINESS LOGIC: This layer only provides utilities and abstractions.

USAGE EXAMPLE:
--------------
    from src.core import BaseTool, CircuitBreakerManager, RAGPipelineException
    
    class MyTool(BaseTool):
        pass

================================================================================
"""

from .base_tool import BaseTool, ToolConfig
from .circuit_breaker import (
    CircuitBreakerManager, 
    CircuitBreakerConfig,
    CircuitState,
)
from .exceptions import (
    RAGPipelineException,
    ValidationError,
    ProcessingError,
    ChunkingError,
    EmbeddingError,
    VectorDBError,
    ConfigurationError,
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolConfig",
    # Circuit breaker
    "CircuitBreakerManager",
    "CircuitBreakerConfig",
    "CircuitState",
    # Exceptions
    "RAGPipelineException",
    "ValidationError",
    "ProcessingError",
    "ChunkingError",
    "EmbeddingError",
    "VectorDBError",
    "ConfigurationError",
]
