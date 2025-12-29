"""
VectorDB Providers package.

Re-exports default providers from individual implementation files.
"""

from src.providers.vectordb.base import IVectorDBProvider
from src.providers.vectordb.qdrant import default_provider as qdrant_default
from src.providers.vectordb.faiss import default_provider as faiss_default

__all__ = [
    "IVectorDBProvider",
    "qdrant_default",
    "faiss_default",
]
