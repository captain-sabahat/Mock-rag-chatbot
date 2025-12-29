"""
FILE: src/providers/llm/base.py

LLM provider interface (contract).
All LLM implementations (Gemini/OpenAI/Anthropic/HF etc.) must implement this.
"""

from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod


class ILLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize provider (setup client, warmup, etc.)."""
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        context: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate an answer from a prompt with optional context."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources."""
        raise NotImplementedError

__all__ = ["ILLMProvider"]