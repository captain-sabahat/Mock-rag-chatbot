"""
FILE: src/providers/embeddings/base.py

Embeddings provider interface (contract).
All embeddings implementations must implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class IEmbeddingsProvider(ABC):
    """Abstract base class for embeddings providers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Load model / warmup / prepare caches."""
        raise NotImplementedError

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Returns:
            List of embeddings; each embedding is a list[float].
        """
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources."""
        raise NotImplementedError
