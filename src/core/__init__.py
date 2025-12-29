# 12 files: Three lines of thought
"""
================================================================================
FILE: src/core/__init__.py
================================================================================

PURPOSE:
    Package initialization for core layer. Exports main handlers and utilities
    for easy imports. Enables clean architecture: from src.core import handlers

WORKFLOW:
    1. Import all handler classes
    2. Export as public API
    3. Enable clean imports throughout codebase

IMPORTS:
    - All handler classes from core modules

OUTPUTS:
    - Handler classes available for import

KEY FACTS:
    - Minimal file (just re-exports)
    - Enables clean package structure
    - Future: add handler factory methods

FUTURE SCOPE (Phase 2+):
    - Add handler factory methods
    - Add handler wiring/composition
    - Add handler registry
    - Add auto-discovery of handlers

TESTING ENVIRONMENT:
    - Import: from src.core import RedisHandler

PRODUCTION DEPLOYMENT:
    - All handlers loaded at startup via dependencies
"""

# ================================================================================
# IMPORTS & EXPORTS
# ================================================================================

from src.core.redis_handler import RedisHandler
from src.core.vector_db_handler import VectorDBHandler
from src.core.embeddings_handler import EmbeddingsHandler
from src.core.llm_handler import LLMHandler
from src.core.slm_handler import SLMHandler
from src.core.doc_ingestion import DocumentIngestionHandler
from src.core.circuit_breaker import CircuitBreaker
from src.core.state_manager import StateManager

__all__ = [
    "RedisHandler",
    "VectorDBHandler",
    "EmbeddingsHandler",
    "LLMHandler",
    "SLMHandler",
    "DocumentIngestionHandler",
    "CircuitBreaker",
    "StateManager"
]
