"""
================================================================================
EMBEDDINGS PACKAGE INITIALIZATION
src/tools/embeddings/__init__.py

MODULE PURPOSE:
───────────────
Package initialization for embedding generation and management.
Exports embedding implementations for vector space representation.

WORKING & METHODOLOGY:
──────────────────────
This module:
1. Initializes embeddings package
2. Exports base classes and implementations
3. Manages version tracking
4. Provides public API for embedding operations

HOW IT CONTRIBUTES TO RAG PIPELINE:
──────────────────────────────────
- Converts text chunks to dense vectors
- Enables semantic similarity search
- Powers vector database indexing
- Supports retrieval-augmented generation

PUBLIC API EXPORTS:
───────────────────
Base Classes & Interfaces:
  • BaseEmbedder: Abstract base for embeddings
  • EmbeddingConfig: Configuration dataclass
  • EmbeddingResult: Embedding output dataclass

Implementations:
  • BaseEmbedder: Abstract interface
  • Mock implementations for testing

EXTERNAL IMPORTS & DEPENDENCIES:
─────────────────────────────────
Internal Submodules:
  • .base_embedder: BaseEmbedder, EmbeddingConfig, EmbeddingResult
  • .bge_embedder: BGEEmbedder (Dense retrieval)
  • .mock_embedder: MockEmbedder (Testing)

External:
  • typing: Type hints
  • logging: Logging support
  • dataclasses: Configuration dataclasses

VERSION TRACKING:
─────────────────
v1.0.0 - Initial release with BGE and mock embedders

================================================================================
"""

__version__ = "1.0.0"

# Critical: Import base classes first
try:
    from .base_embedder import (
        BaseEmbedder,
        EmbeddingConfig,
        EmbeddingResult,
    )
except ImportError as e:
    raise ImportError(f"Failed to import base embedder: {str(e)}")

# Critical: Import BGE embedder
try:
    from .bge_embedder import BGEEmbedder
except ImportError as e:
    raise ImportError(f"Failed to import BGE embedder: {str(e)}")

# Critical: Import mock embedder for testing
try:
    from .mock_embedder import MockEmbedder
except ImportError as e:
    raise ImportError(f"Failed to import mock embedder: {str(e)}")

# Public API exports
__all__ = [
    "BaseEmbedder",
    "EmbeddingConfig",
    "EmbeddingResult",
    "BGEEmbedder",
    "MockEmbedder",
    "__version__",
]

import logging
logger = logging.getLogger(__name__)
logger.info("Embeddings package initialized successfully (v%s)" % __version__)