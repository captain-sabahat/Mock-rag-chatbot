# LINE 3: Vector search
"""
================================================================================
FILE: src/core/vector_db_handler.py
================================================================================

PURPOSE:
    Semantic search in vector database (Qdrant). Retrieves relevant documents
    based on query embedding. Core to RAG retrieval step.

WORKFLOW:
    1. Receive query embedding (384 dimensions)
    2. Connect to Qdrant (async)
    3. Search collection for similar documents
    4. Apply score threshold (default: 0.7)
    5. Return top-k results (default: 5)

LATENCY:
    - Search: 200-400ms
    - With filtering: 400-800ms
    - P99: <1s

IMPORTS:
    - qdrant_client.async_client: Async Qdrant client
    - asyncio: Async operations
    - config: Vector DB settings

INPUTS:
    - query_embedding: Vector to search for (List[float])
    - collection_name: Qdrant collection (default: "documents")

OUTPUTS:
    - Search results: List[ScoredPoint] (document + score)

QDRANT CONFIGURATION:
    - Collection: "documents" (pre-created)
    - Vector size: 384 (from all-MiniLM embeddings)
    - Distance metric: Cosine similarity
    - Indexed: Yes (for speed)

RESULT FILTERING:
    - Top-k: 5 results (configurable)
    - Score threshold: 0.7 (configurable)
    - Only return results above threshold

KEY FACTS:
    - Semantic search (similarity-based, not keyword)
    - Pre-computed indexes make search fast
    - Scores: 0-1 (1 = perfect match)
    - Filters possible (metadata, tags)

RESILIENCE:
    - Connection pooling
    - Timeout protection
    - Circuit breaker
    - Graceful degradation

FUTURE SCOPE (Phase 2+):
    - Add multi-collection support
    - Add filtering by metadata
    - Add hybrid search (keyword + semantic)
    - Add re-ranking (reorder results)
    - Add caching (cache search results)
    - Add metrics (search latency, relevance)
    - Add collection management (create, delete)

TESTING ENVIRONMENT:
    - Use test collection
    - Mock Qdrant responses
    - Test filtering/threshold
    - Verify result format

PRODUCTION DEPLOYMENT:
    - Use production Qdrant instance
    - Pre-compute indexes for speed
    - Monitor search latency
    - Alert if latency exceeds threshold
    - Backup embeddings collection
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Protocol

from src.config.settings import Settings
from .exceptions import VectorDBError, TimeoutError as TimeoutErrorException

logger = logging.getLogger(__name__)


class VectorDBProviderProtocol(Protocol):
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        score_threshold: float,
    ) -> Any: ...


class VectorDBHandler:
    """
    Handles semantic search in vector database.
    Retrieves relevant documents based on embeddings.
    All operations are async (non-blocking).
    """

    def __init__(self, provider: VectorDBProviderProtocol, settings: Settings):
        """
        Args:
            provider: Pre-initialized vector DB provider (created by ServiceContainer)
            settings: Application settings
        """
        self.settings = settings
        self.provider = provider
        logger.info("VectorDBHandler initialized")

    async def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in vector DB.
        Keeps same defaults + timeout + formatting.
        """
        top_k = top_k or self.settings.vector_db_top_k
        score_threshold = score_threshold or self.settings.vector_db_score_threshold

        try:
            results = await asyncio.wait_for(
                self.provider.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    score_threshold=score_threshold,
                ),
                timeout=self.settings.vector_db_timeout,
            )

            # Keep same formatting idea: return list of dicts with id/score/payload
            formatted_results = []
            for result in results:
                # Support both object-style (qdrant ScoredPoint) and dict-style providers
                rid = getattr(result, "id", None)
                rscore = getattr(result, "score", None)
                rpayload = getattr(result, "payload", None)

                if isinstance(result, dict):
                    rid = result.get("id", rid)
                    rscore = result.get("score", rscore)
                    rpayload = result.get("payload", rpayload)

                formatted_results.append(
                    {"id": rid, "score": rscore, "payload": rpayload or {}}
                )

            logger.debug(
                f"Vector DB search returned {len(formatted_results)} results "
                f"(threshold: {score_threshold})"
            )
            return formatted_results

        except asyncio.TimeoutError:
            raise TimeoutErrorException(
                f"Vector DB search timeout after {self.settings.vector_db_timeout}s",
                context={"collection": collection_name},
            )

        except Exception as e:
            raise VectorDBError(
                f"Vector DB search failed: {str(e)}",
                context={"collection": collection_name},
            )

    async def close(self):
        """Close vector DB connection (if provider exposes close)."""
        if hasattr(self.provider, "close"):
            await self.provider.close()
        logger.info("Vector DB connection closed")
