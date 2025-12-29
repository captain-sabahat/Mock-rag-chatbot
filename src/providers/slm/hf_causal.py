"""
FILE: src/providers/slm/hf_causal.py

HuggingFace Causal-LM based SLM provider (e.g., Phi-3).
Defaults are taken from environment variables (Layer 2).

Expected env vars (based on your screenshot):
- ENABLE_SLM=True
- SLM_FOR_SUMMARIZATION=True
- SLM_MODEL=microsoft/phi-3-mini-4k-instruct
- SLM_DEVICE=cpu
- SLM_MAX_TOKENS=512
- SLM_TEMPERATURE=0.5
- SLM_USE_CACHE=True
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

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
class HFCausalSLMConfig:
    enabled: bool
    for_summarization: bool
    model: str
    device: str
    max_tokens_default: int
    temperature_default: float
    use_cache: bool


class HFCausalProvider(ISLMProvider):
    """
    SLM summarization provider using a causal LM.

    Implementation:
    - Uses `transformers` text-generation pipeline.
    - Runs generation in a thread (pipeline is blocking).
    """

    def __init__(self, config: HFCausalSLMConfig):
        self.config = config
        self._pipe = None
        logger.info(
            "HFCausalProvider created: enabled=%s model=%s device=%s use_cache=%s",
            self.config.enabled,
            self.config.model,
            self.config.device,
            self.config.use_cache,
        )

    async def initialize(self) -> None:
        if not self.config.enabled:
            logger.info("SLM disabled via ENABLE_SLM=False (skipping init).")
            return

        try:
            from transformers import pipeline

            def _load():
                # `device` for pipeline:
                # - CPU: device=-1
                # - CUDA: device=0 (or specific GPU index)
                device = -1
                if self.config.device.lower() in {"cuda", "gpu"}:
                    device = 0

                return pipeline(
                    task="text-generation",
                    model=self.config.model,
                    device=device,
                )

            self._pipe = await asyncio.to_thread(_load)
            logger.info("✓ HFCausalProvider initialized: %s", self.config.model)
        except Exception as e:
            logger.error("HFCausalProvider init failed: %s", str(e), exc_info=True)
            raise

    async def summarize(
        self,
        text: str,
        max_tokens: int = 256,
        temperature: float = 0.5,
    ) -> str:
        if not self.config.enabled:
            raise RuntimeError("SLM is disabled (ENABLE_SLM=False).")

        if not self.config.for_summarization:
            raise RuntimeError("SLM_FOR_SUMMARIZATION=False; summarization is disabled.")

        if self._pipe is None:
            raise RuntimeError("HFCausalProvider not initialized. Call initialize() first.")

        # Keep prompt simple and deterministic enough for summarization
        max_tokens = int(max_tokens or self.config.max_tokens_default)
        temperature = float(temperature if temperature is not None else self.config.temperature_default)

        # Prevent super long docs from exploding prompt size
        if len(text) > 6000:
            text = text[:6000] + "..."

        prompt = (
            "Summarize the following text clearly and concisely.\n\n"
            f"TEXT:\n{text}\n\n"
            "SUMMARY:"
        )

        def _run() -> str:
            out = self._pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                return_full_text=False,
            )
            # pipeline returns list[dict], usually [{"generated_text": "..."}]
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return (out[0].get("generated_text") or "").strip()
            return str(out).strip()

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.error("HFCausalProvider summarize failed: %s", str(e), exc_info=True)
            raise

    async def shutdown(self) -> None:
        self._pipe = None
        logger.info("HFCausalProvider shutdown complete")


# =============================================================================
# Layer-2 defaults (from env) + exported instance
# =============================================================================

DEFAULT_CONFIG = HFCausalSLMConfig(
    enabled=_env_bool("ENABLE_SLM", True),
    for_summarization=_env_bool("SLM_FOR_SUMMARIZATION", True),
    model=os.getenv("SLM_MODEL", "microsoft/phi-3-mini-4k-instruct"),
    device=os.getenv("SLM_DEVICE", "cpu"),
    max_tokens_default=_env_int("SLM_MAX_TOKENS", 512),
    temperature_default=_env_float("SLM_TEMPERATURE", 0.5),
    use_cache=_env_bool("SLM_USE_CACHE", True),
)

default_provider = HFCausalProvider(config=DEFAULT_CONFIG)

__all__ = ["HFCausalSLMConfig", "HFCausalProvider", "DEFAULT_CONFIG", "default_provider"]
