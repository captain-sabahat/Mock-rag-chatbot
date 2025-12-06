"""
================================================================================
VECTOR DATABASE PACKAGE INITIALIZATION
src/tools/vectordb/__init__.py

MODULE PURPOSE:
───────────────
Package initialization for vector database management.
Exports vector store and client implementations.

WORKING & METHODOLOGY:
──────────────────────
This module:
1. Initializes vectordb package
2. Exports base classes and implementations
3. Manages version tracking
4. Provides public API for vector operations

HOW IT CONTRIBUTES TO RAG PIPELINE:
──────────────────────────────────
- Stores embedding vectors efficiently
- Enables semantic similarity search
- Powers retrieval ranking
- Supports vector indexing

PUBLIC API EXPORTS:
───────────────────
Base Classes & Interfaces:
  • BaseVectorDB: Abstract base for vector stores
  • VectorStoreConfig: Configuration dataclass
  • SearchResult: Search output dataclass

Implementations:
  • FAISSClient: FAISS vector store client
  • MockVectorDB: Mock implementation for testing

VERSION TRACKING:
─────────────────
v1.0.0 - Initial release with FAISS and mock clients

================================================================================
"""

__version__ = "1.0.0"

# Critical: Import base classes first
try:
    from .base_vectordb import (
        BaseVectorDB,
        VectorStoreConfig,
        SearchResult,
    )
except ImportError as e:
    raise ImportError(f"Failed to import base vectordb: {str(e)}")

# Critical: Import FAISS client
try:
    from .faiss_client import FAISSClient
except ImportError as e:
    raise ImportError(f"Failed to import FAISS client: {str(e)}")

# Critical: Import mock vectordb
try:
    from .mock_vectordb import MockVectorDB
except ImportError as e:
    raise ImportError(f"Failed to import mock vectordb: {str(e)}")

# Public API exports
__all__ = [
    "BaseVectorDB",
    "VectorStoreConfig",
    "SearchResult",
    "FAISSClient",
    "MockVectorDB",
    "__version__",
]

import logging
logger = logging.getLogger(__name__)
logger.info("Vector database package initialized successfully (v%s)" % __version__)
