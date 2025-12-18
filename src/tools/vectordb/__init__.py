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
from .vectordb_registry import (
    VectorDBConfig,
    VectorStoreResult,
    BaseVectorDB,
    FAISSVectorDB,
    QdrantVectorDB,
    VectorDBFactory,
    load_vectordb_config,
)

# Register backends
VectorDBFactory.register('faiss', FAISSVectorDB)
VectorDBFactory.register('qdrant', QdrantVectorDB)

__all__ = [
    'VectorDBConfig',
    'VectorStoreResult',
    'VectorDBFactory',
    'load_vectordb_config',
]
