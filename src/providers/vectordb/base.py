"""
FILE: src/providers/vectordb/base.py

VectorDB provider interface (contract).
All vector database implementations must implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IVectorDBProvider(ABC):
    """Abstract base class for vector database providers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize provider (connect/load index/etc.)."""
        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert/update vectors with optional payload metadata."""
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Similarity search.

        Returns list of:
          { "id": str, "score": float, "payload": dict }
        """
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources."""
        raise NotImplementedError
