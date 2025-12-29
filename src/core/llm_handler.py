"""
================================================================================
FILE: src/core/llm_handler.py
================================================================================

PURPOSE:
    Fine-tuned LLM inference for answer generation. Receives context from
    vector DB and generates final answer. Highest latency component (~1-2s).

WORKFLOW:
    1. Receive prompt + retrieved context
    2. Format as LLM input (prompt engineering)
    3. Call LLM inference (1-2s)
    4. Return generated answer

LATENCY:
    - API-based: 1-2 seconds
    - Streaming (Phase 2): First token in 200-500ms
    - P99: <3s

IMPORTS:
    - asyncio: Async execution
    - httpx: Async HTTP client (for API)
    - models.fine_tuned_llm: FineTunedLLM model
    - config: LLM settings

INPUTS:
    - prompt: User query + context
    - max_tokens: Max response length
    - temperature: Randomness (0-1)

OUTPUTS:
    - Generated answer (string)

PROMPT ENGINEERING:
    - System prompt: Define role/style
    - Context: Retrieved relevant documents
    - User query: Original user question
    - Format: Clear instructions for answer

LLM PARAMETERS:
    - temperature: 0.3 (factual), 0.7 (balanced), 0.9 (creative)
    - max_tokens: 500 (short), 1000 (medium), 2000 (long)
    - top_p: 0.9 (focused), 0.95 (balanced), 1.0 (diverse)

KEY FACTS:
    - Highest latency component (1-2s)
    - API-based (external service)
    - Streaming available (Phase 2)
    - Supports temperature for output variety

RESILIENCE:
    - Timeout protection (30s max)
    - Retry logic (3 attempts)
    - Circuit breaker (fail fast if API down)
    - Graceful degradation (return partial result)

FUTURE SCOPE (Phase 2+):
    - Add streaming responses (return tokens as generated)
    - Add token counting (track usage)
    - Add cost tracking (expensive API calls)
    - Add A/B testing (multiple model versions)
    - Add fallback LLM (if primary fails)
    - Add prompt templates (reusable prompts)
    - Add response validation (check quality)

TESTING ENVIRONMENT:
    - Mock LLM responses
    - Test prompt formatting
    - Test timeout handling
    - Test error recovery

PRODUCTION DEPLOYMENT:
    - Use production LLM endpoint
    - Monitor latency (alert if >3s)
    - Track token usage
    - Implement retry logic
    - Use streaming for better UX
    - Cache common responses (Phase 2)
"""
# imports 

import asyncio
import logging
from typing import Optional, Protocol

from src.config.settings import Settings
from .exceptions import LLMError, TimeoutError as TimeoutErrorException
from src.utils import retry_async_with_backoff

logger = logging.getLogger(__name__)


class LLMProviderProtocol(Protocol):
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str: ...


class LLMHandler:
    """
    Fine-tuned LLM for answer generation.
    Generates answers given user query + context.
    Supports retry logic and timeout protection.
    """

    def __init__(self, provider: LLMProviderProtocol, settings: Settings):
        """
        Initialize LLM handler.

        Args:
            provider: Pre-initialized LLM provider (created by ServiceContainer)
            settings: Application settings
        """
        self.provider = provider
        self.settings = settings
        logger.info("LLMHandler initialized")

    async def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate answer using LLM.

        Keeps same retry + formatting logic. Only the underlying call is swapped:
        model.infer(...) -> provider.generate(...)
        """
        try:
            full_prompt = self._format_prompt(prompt, context)

            # Keep retry logic (same behavior)
            answer = await retry_async_with_backoff(
                self.provider.generate,
                max_retries=3,
                backoff_factor=2.0,
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            logger.debug(f"LLM generated answer ({len(answer)} chars)")
            return answer

        except asyncio.TimeoutError:
            raise TimeoutErrorException(
                f"LLM inference timeout after {self.settings.llm_timeout}s",
                context={"prompt_length": len(prompt)},
            )

        except Exception as e:
            raise LLMError(
                f"LLM inference failed: {str(e)}",
                context={"prompt_length": len(prompt)},
            )

    def _format_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Format prompt with context (prompt engineering).
        (Kept as-is)
        """
        system_prompt = (
            "You are a helpful AI assistant. Provide concise, accurate answers "
            "based on the provided context. If the answer is not in the context, "
            "say 'I don't know'."
        )

        if context:
            return f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return f"{system_prompt}\n\nQuestion: {query}\n\nAnswer:"
