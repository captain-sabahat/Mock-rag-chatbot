"""
FILE: src/providers/llm/huggingface.py

HuggingFace LLM provider (local causal LM) using transformers.

Layer-2 defaults live here (DEFAULT_*). ServiceContainer imports:
  from src.providers.llm.huggingface import default_provider
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

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
class HFLLMConfig:
    model: str
    device: str = "auto"  # "cpu" | "cuda" | "auto"
    max_new_tokens_default: int = 512
    temperature_default: float = 0.7
    quantization: Optional[str] = None  # "int8" | "int4" | None


class HuggingFaceProvider(ILLMProvider):
    """
    Local LLM using transformers AutoModelForCausalLM.

    Notes:
    - Generation is blocking; run it in a thread to keep async pipeline responsive.
    - For GPU quantization, bitsandbytes may be required.
    """

    def __init__(self, config: HFLLMConfig):
        self.config = config
        self._tokenizer = None
        self._model = None
        logger.info(
            "HuggingFaceProvider created (model=%s device=%s quant=%s)",
            self.config.model,
            self.config.device,
            self.config.quantization,
        )

    async def initialize(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            def _load():
                tok = AutoTokenizer.from_pretrained(self.config.model)
                kwargs = {"device_map": self.config.device}

                if self.config.quantization == "int8":
                    kwargs["load_in_8bit"] = True
                elif self.config.quantization == "int4":
                    kwargs["load_in_4bit"] = True

                mdl = AutoModelForCausalLM.from_pretrained(self.config.model, **kwargs)
                return tok, mdl

            self._tokenizer, self._model = await asyncio.to_thread(_load)
            logger.info("✓ HuggingFace LLM initialized (model=%s)", self.config.model)
        except Exception as e:
            logger.error("HF LLM init failed: %s", str(e), exc_info=True)
            raise

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("HuggingFaceProvider not initialized. Call initialize() first.")

        def _call() -> str:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            # Move tensors to model device if needed
            try:
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            except Exception:
                pass

            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                do_sample=True if float(temperature) > 0 else False,
            )

            text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Optional: strip the prompt part for cleaner response
            if text.startswith(prompt):
                return text[len(prompt) :].lstrip()
            return text

        try:
            return await asyncio.to_thread(_call)
        except Exception as e:
            logger.error("HF LLM generate failed: %s", str(e), exc_info=True)
            raise

    async def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None
        logger.info("HuggingFaceProvider shutdown complete")


# =============================================================================
# Layer-2 defaults
# =============================================================================

DEFAULT_MODEL = os.getenv("HF_LLM_MODEL", "gpt2")
DEFAULT_DEVICE = os.getenv("HF_LLM_DEVICE", "auto")
DEFAULT_MAX_NEW_TOKENS = _env_int("LLM_MAX_TOKENS", 512)
DEFAULT_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.7)
DEFAULT_QUANTIZATION = os.getenv("HF_LLM_QUANTIZATION", "").strip() or None  # int8/int4/None

DEFAULT_CONFIG = HFLLMConfig(
    model=DEFAULT_MODEL,
    device=DEFAULT_DEVICE,
    max_new_tokens_default=DEFAULT_MAX_NEW_TOKENS,
    temperature_default=DEFAULT_TEMPERATURE,
    quantization=DEFAULT_QUANTIZATION,
)

default_provider = HuggingFaceProvider(config=DEFAULT_CONFIG)

__all__ = ["HFLLMConfig", "HuggingFaceProvider", "DEFAULT_CONFIG", "default_provider"]
