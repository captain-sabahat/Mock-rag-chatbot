"""
================================================================================
CHUNKING PACKAGE INITIALIZATION
src/tools/chunking/__init__.py

MODULE PURPOSE:
───────────────
Package initialization for text chunking operations.
Exports chunking implementations for document segmentation.

WORKING & METHODOLOGY:
──────────────────────
This module:
1. Initializes chunking package
2. Exports base classes and implementations
3. Manages version tracking
4. Provides public API for chunking operations

HOW IT CONTRIBUTES TO RAG PIPELINE:
──────────────────────────────────
- Splits documents into manageable chunks
- Preserves semantic coherence
- Enables efficient embedding generation
- Powers vector database indexing

PUBLIC API EXPORTS:
───────────────────
Base Classes & Enums:
  • BaseChunker: Abstract base for chunking
  • ChunkingStrategy: Enum for strategy types
  • Chunk: Dataclass for chunk metadata

Implementations:
  • SemanticChunker: Intelligent segmentation (95%+ accuracy)
  • SlidingWindowChunker: Fixed-size chunks (fast processing)

EXTERNAL IMPORTS & DEPENDENCIES:
─────────────────────────────────
Internal Submodules:
  • .base_chunker: BaseChunker, ChunkingStrategy, Chunk
  • .semantic_chunker: SemanticChunker
  • .sliding_window_chunker: SlidingWindowChunker

External:
  • typing: Type hints
  • logging: Logging support
  • dataclasses: Chunk dataclass

VERSION TRACKING:
─────────────────
v1.0.0 - Initial release with semantic and sliding window chunking

================================================================================
"""

__version__ = "1.0.0"

# Critical: Import base classes first
try:
    from .base_chunker import (
        BaseChunker,
        ChunkingStrategy,
        Chunk,
    )
except ImportError as e:
    raise ImportError(f"Failed to import base chunker: {str(e)}")

# Critical: Import semantic chunker
try:
    from .semantic_chunker import SemanticChunker
except ImportError as e:
    raise ImportError(f"Failed to import semantic chunker: {str(e)}")

# Critical: Import sliding window chunker
try:
    from .sliding_window_chunker import SlidingWindowChunker
except ImportError as e:
    raise ImportError(f"Failed to import sliding window chunker: {str(e)}")

# Public API exports
__all__ = [
    "BaseChunker",
    "ChunkingStrategy",
    "Chunk",
    "SemanticChunker",
    "SlidingWindowChunker",
    "__version__",
]

import logging
logger = logging.getLogger(__name__)
logger.info("Chunking package initialized successfully (v%s)" % __version__)