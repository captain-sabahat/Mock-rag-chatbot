"""
FILE: src/providers/llm/gemini.py

Gemini LLM provider using the new google-genai SDK.

Layer-2 defaults live here (DEFAULT_*). ServiceContainer imports:

from src.providers.llm.gemini import default_provider
"""

from __future__ import annotations

import asyncio
import logging
import os

from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types as genai_types  # for GenerateContentConfig

from .base import ILLMProvider

logger = logging.getLogger(__name__)

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return float(v)

@dataclass(frozen=True)
class GeminiLLMConfig:
    model: str
    api_key_env: str = "GEMINI_API_KEY"
    timeout_s: int = 30
    default_max_tokens: int = 1024
    default_temperature: float = 0.7

class GeminiProvider(ILLMProvider):
    """
    Gemini provider using the new google-genai SDK.

    Notes:
    - google-genai client is sync; generate() runs in a thread
      to avoid blocking the event loop.
    - Supports RAG context injection for augmented generation.
    """

    def __init__(self, config: GeminiLLMConfig) -> None:
        self.config = config
        self._client: Optional[genai.Client] = None
        logger.info("GeminiProvider created (model=%s)", self.config.model)

    async def initialize(self) -> None:
        """
        Initialize Gemini client.

        Uses google-genai Client(api_key=...) instead of the old
        google-generativeai genai.configure(...) pattern.
        """
        try:
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise ValueError(f"{self.config.api_key_env} not set")

            # New SDK: create a Client with api_key
            self._client = genai.Client(api_key=api_key)
            logger.info("✓ Gemini initialized (model=%s)", self.config.model)

        except Exception as e:
            logger.error("Gemini init failed: %s", str(e), exc_info=True)
            raise

    async def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text from a prompt using Gemini with optional RAG context.

        Runs the sync google-genai call in a worker thread and
        applies an overall timeout via asyncio.wait_for.
        """
        if self._client is None:
            raise RuntimeError("GeminiProvider not initialized. Call initialize() first.")

        # Prepare effective params, falling back to config defaults
        effective_max_tokens = int(max_tokens or self.config.default_max_tokens)
        effective_temperature = float(temperature or self.config.default_temperature)

        # Build augmented prompt with context
        full_prompt = self._build_prompt(prompt, context)

        def _call() -> str:
            # google-genai pattern: client.models.generate_content(...)
            resp = self._client.models.generate_content(
                model=self.config.model,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=effective_temperature,
                    max_output_tokens=effective_max_tokens,
                ),
            )

            # resp.text is the simplest accessor
            return getattr(resp, "text", "") or ""

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_call),
                timeout=self.config.timeout_s,
            )

        except asyncio.TimeoutError:
            raise TimeoutError(f"Gemini timed out after {self.config.timeout_s}s")

        except Exception as e:
            logger.error("Gemini generate failed: %s", str(e), exc_info=True)
            raise

    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Build augmented prompt with context for RAG.
        
        If context is provided, prepends it to the prompt.
        Otherwise returns prompt as-is.
        """
        if not context:
            return prompt
        
        return f"""You are a helpful assistant. Use the provided context to answer the question.

CONTEXT:
{context}

QUESTION:
{prompt}

ANSWER:"""

    async def shutdown(self) -> None:
        """
        Shutdown provider.

        google-genai does not require explicit close; just drop the client.
        """
        self._client = None
        logger.info("GeminiProvider shutdown complete")

# =============================================================================
# Layer-2 defaults (edit these to change model/config without touching container)
# =============================================================================

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_TIMEOUT_S = _env_int("LLM_TIMEOUT", 30)
DEFAULT_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 1024)
DEFAULT_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.7)

DEFAULT_CONFIG = GeminiLLMConfig(
    model=DEFAULT_MODEL,
    timeout_s=DEFAULT_TIMEOUT_S,
    default_max_tokens=DEFAULT_MAX_TOKENS,
    default_temperature=DEFAULT_TEMPERATURE,
)

default_provider = GeminiProvider(config=DEFAULT_CONFIG)

__all__ = ["GeminiLLMConfig", "GeminiProvider", "DEFAULT_CONFIG", "default_provider"]
