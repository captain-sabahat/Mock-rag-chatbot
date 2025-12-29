"""
FILE: src/providers/slm/base.py

SLM provider interface (contract).
All SLM implementations must implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ISLMProvider(ABC):
    """Abstract base class for small language model providers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize provider (load model, tokenizer, etc.)."""
        raise NotImplementedError

    @abstractmethod
    async def summarize(
        self,
        text: str,
        max_tokens: int = 256,
        temperature: float = 0.5,
    ) -> str:
        """Summarize text into a shorter response."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources."""
        raise NotImplementedError
