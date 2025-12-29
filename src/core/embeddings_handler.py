# LINE 3: Query embedding
"""
================================================================================
FILE: src/core/embeddings_handler.py
================================================================================

PURPOSE:
    Generates embeddings for query strings and documents. Critical for semantic
    search in RAG pipeline. Supports batch processing for latency optimization.

LATENCY:
    - CPU: 100-500ms per query
    - GPU: 10-50ms per query
    - Batch (32): 200ms total (vs 3.2s individual)

WORKFLOW:
    1. Receive text to embed (query or document chunk)
    2. Use embeddings model (all-MiniLM-L6-v2)
    3. Generate embedding vector (384 dimensions)
    4. Return embedding for vector DB search

IMPORTS:
    - asyncio: Async execution
    - models.embeddings_model: EmbeddingsModel class
    - config: Batch size, timeout config

INPUTS:
    - texts: List of text strings to embed
    - Model: Pre-initialized SentenceTransformer

OUTPUTS:
    - List of embedding vectors (List[List[float]])

OPTIMIZATION:
    - Batch processing: Process multiple queries together
    - GPU acceleration: 10x faster than CPU
    - Mixed precision: float16 on GPU (faster, less precise)
    - Connection pooling: Reuse model across requests

BATCH PROCESSING STRATEGY:
    - Collect queries in queue
    - Wait max 10ms or until batch full (32)
    - Encode all together (faster than individual)
    - Return embeddings

DEVICE SUPPORT:
    - CPU: float32, always available
    - CUDA: float16 mixed precision, 10x faster
    - Metal (Apple Silicon): float16, 5x faster
    - NPU: Custom implementation

KEY FACTS:
    - Embeddings are deterministic (same text = same vector)
    - Embeddings are expensive (100-500ms)
    - Batch processing reduces per-query cost
    - Device abstraction enables CPU/GPU switching

FUTURE SCOPE (Phase 2+):
    - Add embedding caching (don't re-embed same text)
    - Add batch queuing (collect + encode periodically)
    - Add dynamic batch sizing (adaptive to load)
    - Add quantization (reduce memory, smaller vectors)
    - Add incremental embeddings (update instead of replace)
    - Add embedding metrics (latency, throughput)

TESTING ENVIRONMENT:
    - Use tiny embeddings model for speed
    - Mock embeddings in unit tests
    - Test batch processing
    - Verify deterministic output

PRODUCTION DEPLOYMENT:
    - Use full embeddings model (384 dimensions)
    - Enable GPU acceleration (if available)
    - Monitor latency (alert if >500ms)
    - Batch size: 32-128 depending on GPU memory
    - Cache embeddings for repeated texts
"""

import asyncio
import logging
from typing import List, Protocol

from src.config.settings import Settings
from .exceptions import EmbeddingError, TimeoutError as TimeoutErrorException

logger = logging.getLogger(__name__)


class EmbeddingsProviderProtocol(Protocol):
    async def embed_texts(self, texts: List[str]) -> List[List[float]]: ...


class EmbeddingsHandler:
    """
    Handles embeddings generation for queries and documents.
    Supports timeout protection.
    """

    def __init__(self, provider: EmbeddingsProviderProtocol, settings: Settings):
        """
        Args:
            provider: Pre-initialized embeddings provider (created by ServiceContainer)
            settings: Application settings
        """
        self.provider = provider
        self.settings = settings
        logger.info("EmbeddingsHandler initialized")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Keeps same timeout + error behavior.
        Only underlying call is swapped:
        model.infer(...) -> provider.embed_texts(...)
        """
        try:
            embeddings = await asyncio.wait_for(
                self.provider.embed_texts(texts),
                timeout=self.settings.embeddings_timeout,
            )

            logger.debug(
                f"Embeddings generated: {len(texts)} texts → {len(embeddings)} vectors"
            )
            return embeddings

        except asyncio.TimeoutError:
            raise TimeoutErrorException(
                f"Embeddings timeout after {self.settings.embeddings_timeout}s",
                context={"text_count": len(texts)},
            )

        except Exception as e:
            raise EmbeddingError(
                f"Embedding generation failed: {str(e)}",
                context={"text_count": len(texts)},
            )

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for single query.
        (Kept as-is)
        """
        embeddings = await self.embed_texts([query])
        return embeddings[0]
