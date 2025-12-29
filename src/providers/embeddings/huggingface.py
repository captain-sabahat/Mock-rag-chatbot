"""
FILE: src/providers/embeddings/huggingface.py

HuggingFace embeddings provider using sentence-transformers.

Reads defaults from environment (Layer 2 defaults live here).
The container should import: `default_provider` from this file.

Expected env vars (as per your screenshot):
- EMBEDDINGS_MODEL=BAAI/bge-small-en-v1.5
- EMBEDDINGS_DEVICE=cpu
- EMBEDDINGS_BATCH_SIZE=32
- EMBEDDINGS_NORMALIZE=True
- EMBEDDINGS_CACHE_DIR=./cache/embeddings
- EMBEDDINGS_DIMENSION=384
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from .base import IEmbeddingsProvider

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return int(val)


@dataclass(frozen=True)
class HFEmbeddingsConfig:
    model_name: str
    device: str
    batch_size: int
    normalize: bool
    cache_dir: str
    dimension: int


class HFEmbeddingsProvider(IEmbeddingsProvider):
    """
    SentenceTransformer-based embeddings provider.

    Notes:
    - Uses asyncio.to_thread because SentenceTransformer.encode is blocking.
    - Normalization is done via sentence-transformers `normalize_embeddings=True`
      (preferred for speed/consistency).
    """

    def __init__(self, config: HFEmbeddingsConfig):
        self.config = config
        self._model = None  # SentenceTransformer
        logger.info(
            "HFEmbeddingsProvider created: model=%s device=%s batch=%s normalize=%s dim=%s cache_dir=%s",
            self.config.model_name,
            self.config.device,
            self.config.batch_size,
            self.config.normalize,
            self.config.dimension,
            self.config.cache_dir,
        )

    async def initialize(self) -> None:
        try:
            os.makedirs(self.config.cache_dir, exist_ok=True)

            # sentence-transformers uses HF_HOME / TRANSFORMERS_CACHE; set a local cache if desired
            # (optional but helpful for server deployments)
            os.environ.setdefault("HF_HOME", self.config.cache_dir)
            os.environ.setdefault("TRANSFORMERS_CACHE", self.config.cache_dir)

            from sentence_transformers import SentenceTransformer

            # Loading can be slow; keep it off the event loop
            self._model = await asyncio.to_thread(
                SentenceTransformer,
                self.config.model_name,
                device=self.config.device,
            )

            # Optional: dimension sanity check (if model exposes it)
            try:
                emb_dim = int(self._model.get_sentence_embedding_dimension())
                if self.config.dimension and emb_dim != int(self.config.dimension):
                    logger.warning(
                        "Embeddings dimension mismatch: env=%s model=%s",
                        self.config.dimension,
                        emb_dim,
                    )
            except Exception:
                # Not fatal; different ST versions may behave differently
                pass

            logger.info("✓ HFEmbeddingsProvider initialized")
        except Exception as e:
            logger.error("HFEmbeddingsProvider init failed: %s", str(e), exc_info=True)
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors, one per input text
            
        Raises:
            RuntimeError: If provider not initialized
            TypeError: If input types are invalid
            ValueError: If dimension mismatch occurs
        """
        if self._model is None:
            raise RuntimeError("HFEmbeddingsProvider not initialized. Call initialize() first.")

        if not texts:
            return []

        # Validate input types
        if not isinstance(texts, list):
            raise TypeError(f"texts must be List[str], got {type(texts)}")
        
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("All items in texts must be strings")

        # SentenceTransformer.encode is blocking, run in thread pool
        def _encode() -> List[List[float]]:
            vectors = self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
            )
            return vectors.tolist()

        try:
            embeddings = await asyncio.to_thread(_encode)

            # Validate dimension if configured
            if self.config.dimension:
                for idx, v in enumerate(embeddings):
                    if len(v) != self.config.dimension:
                        raise ValueError(
                            f"Embedding dimension mismatch at index={idx}: "
                            f"got={len(v)} expected={self.config.dimension}"
                        )

            logger.debug(
                "Generated %d embeddings, dimension=%d",
                len(embeddings),
                len(embeddings[0]) if embeddings else 0
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(
                "HFEmbeddingsProvider embed_texts failed: input_count=%d, error=%s",
                len(texts),
                str(e),
                exc_info=True
            )
            raise


    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        Convenience wrapper for embed_texts with single input.
        
        Args:
            text: Single text to embed
            
        Returns:
            Single embedding vector as List[float]
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If provider not initialized
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        embeddings = await self.embed_texts([text])
        return embeddings[0]

        async def shutdown(self) -> None:
            # sentence-transformers doesn't require explicit close; release reference for GC
            self._model = None
            logger.info("HFEmbeddingsProvider shutdown complete")

    async def shutdown(self) -> None:
        """Shutdown the embeddings provider and release resources."""
        if self._model is not None:
            logger.info("Shutting down HFEmbeddingsProvider...")
            del self._model
            self._model = None
            logger.info("✓ HFEmbeddingsProvider shutdown complete")
            
# =============================================================================
# Layer 2 defaults (from env) + exported instance
# =============================================================================

DEFAULT_CONFIG = HFEmbeddingsConfig(
    model_name=os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5"),
    device=os.getenv("EMBEDDINGS_DEVICE", "cpu"),
    batch_size=_env_int("EMBEDDINGS_BATCH_SIZE", 32),
    normalize=_env_bool("EMBEDDINGS_NORMALIZE", True),
    cache_dir=os.getenv("EMBEDDINGS_CACHE_DIR", "./cache/embeddings"),
    dimension=_env_int("EMBEDDINGS_DIMENSION", 384),
)

default_provider = HFEmbeddingsProvider(config=DEFAULT_CONFIG)

__all__ = ["HFEmbeddingsProvider", "HFEmbeddingsConfig", "DEFAULT_CONFIG", "default_provider"]
