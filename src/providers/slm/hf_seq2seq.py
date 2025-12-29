"""
FILE: src/providers/slm/hf_seq2seq.py

HuggingFace Seq2Seq summarization provider (e.g., DistilBART).
This is usually better for summarization than causal LM.

Env vars:
- ENABLE_SLM
- SLM_FOR_SUMMARIZATION
- SLM_SEQ2SEQ_MODEL (optional; defaults to distilbart-cnn)
- SLM_DEVICE
- SLM_MAX_TOKENS
- SLM_TEMPERATURE
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

from .base import ISLMProvider

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


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
class HFSeq2SeqSLMConfig:
    enabled: bool
    for_summarization: bool
    model: str
    device: str
    max_tokens_default: int
    temperature_default: float


class HFSeq2SeqProvider(ISLMProvider):
    """
    Summarization provider using transformers summarization pipeline.
    """

    def __init__(self, config: HFSeq2SeqSLMConfig):
        self.config = config
        self._pipe = None
        logger.info("HFSeq2SeqProvider created: enabled=%s model=%s", self.config.enabled, self.config.model)

    async def initialize(self) -> None:
        if not self.config.enabled:
            logger.info("SLM disabled via ENABLE_SLM=False (skipping init).")
            return

        try:
            from transformers import pipeline

            def _load():
                device = -1
                if self.config.device.lower() in {"cuda", "gpu"}:
                    device = 0

                return pipeline(
                    task="summarization",
                    model=self.config.model,
                    device=device,
                )

            self._pipe = await asyncio.to_thread(_load)
            logger.info("✓ HFSeq2SeqProvider initialized: %s", self.config.model)
        except Exception as e:
            logger.error("HFSeq2SeqProvider init failed: %s", str(e), exc_info=True)
            raise

    async def summarize(self, text: str, max_tokens: int = 256, temperature: float = 0.5) -> str:
        if not self.config.enabled:
            raise RuntimeError("SLM is disabled (ENABLE_SLM=False).")

        if not self.config.for_summarization:
            raise RuntimeError("SLM_FOR_SUMMARIZATION=False; summarization is disabled.")

        if self._pipe is None:
            raise RuntimeError("HFSeq2SeqProvider not initialized. Call initialize() first.")

        # Pipelines want input lengths under control
        if len(text) > 9000:
            text = text[:9000] + "..."

        max_tokens = int(max_tokens or self.config.max_tokens_default)
        # Some summarization pipelines ignore temperature; keep for API consistency
        temperature = float(temperature if temperature is not None else self.config.temperature_default)

        def _run() -> str:
            out = self._pipe(
                text,
                max_length=max_tokens,
                min_length=max(20, min(80, max_tokens // 3)),
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
            )
            # out is usually [{"summary_text": "..."}]
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return (out[0].get("summary_text") or "").strip()
            return str(out).strip()

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.error("HFSeq2SeqProvider summarize failed: %s", str(e), exc_info=True)
            raise

    async def shutdown(self) -> None:
        self._pipe = None
        logger.info("HFSeq2SeqProvider shutdown complete")


# =============================================================================
# Layer-2 defaults + exported instance
# =============================================================================

DEFAULT_CONFIG = HFSeq2SeqSLMConfig(
    enabled=_env_bool("ENABLE_SLM", True),
    for_summarization=_env_bool("SLM_FOR_SUMMARIZATION", True),
    model=os.getenv("SLM_SEQ2SEQ_MODEL", "sshleifer/distilbart-cnn-12-6"),
    device=os.getenv("SLM_DEVICE", "cpu"),
    max_tokens_default=_env_int("SLM_MAX_TOKENS", 256),
    temperature_default=_env_float("SLM_TEMPERATURE", 0.5),
)

default_provider = HFSeq2SeqProvider(config=DEFAULT_CONFIG)

__all__ = ["HFSeq2SeqSLMConfig", "HFSeq2SeqProvider", "DEFAULT_CONFIG", "default_provider"]
