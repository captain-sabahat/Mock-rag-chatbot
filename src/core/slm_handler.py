"""
================================================================================
FILE: src/core/slm_handler.py
================================================================================
PURPOSE:
SLM summarization handler.
(Option A) Tool/provider is created in ServiceContainer and injected here.
All pipeline logic remains in this file.
================================================================================
"""

import asyncio
import logging
from typing import Protocol

from src.config.settings import Settings
from .exceptions import SLMError, SLMTimeoutError

logger = logging.getLogger(__name__)


class SLMProviderProtocol(Protocol):
    async def summarize(self, text: str, max_length: int = 150) -> str: ...


class SLMHandler:
    """
    Small Language Model for document summarization.
    Supports timeout protection.
    """

    def __init__(self, provider: SLMProviderProtocol, settings: Settings):
        """
        Args:
            provider: Pre-initialized SLM provider (created by ServiceContainer)
            settings: Application settings
        """
        self.provider = provider
        self.settings = settings
        logger.info("SLMHandler initialized")

    async def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50,
    ) -> str:
        """
        Summarize document text using SLM.

        Keeps same truncation + timeout behavior.
        Only underlying call is swapped:
        model.infer(...) -> provider.summarize(...)
        """
        try:
            if len(text) > 5000:
                text = text[:5000] + "..."
                logger.warning("Document truncated to 5000 chars")

            summary = await asyncio.wait_for(
                self.provider.summarize(text, max_length=max_length),
                timeout=self.settings.slm_timeout,
            )

            logger.debug(f"SLM summary generated ({len(summary)} chars)")
            return summary

        except asyncio.TimeoutError:
            raise SLMTimeoutError(
                f"SLM summarization timeout after {self.settings.slm_timeout}s",
                context={"text_length": len(text)},
            )

        except Exception as e:
            raise SLMError(
                f"SLM summarization failed: {str(e)}",
                context={"text_length": len(text)},
            )
