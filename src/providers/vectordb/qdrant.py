"""
FILE: src/providers/vectordb/qdrant.py

Qdrant provider (optional alternative to FAISS).

Env vars (common):
- QDRANT_URL=http://localhost:6333
- QDRANT_API_KEY=... (optional)
- QDRANT_COLLECTION=documents (optional default)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import IVectorDBProvider

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)


@dataclass(frozen=True)
class QdrantConfig:
    url: str
    api_key: Optional[str]
    default_collection: str
    timeout_s: int


class QdrantProvider(IVectorDBProvider):
    def __init__(self, config: QdrantConfig):
        self.config = config
        self._client = None
        logger.info("QdrantProvider created: url=%s", self.config.url)

    async def initialize(self) -> None:
        try:
            from qdrant_client.async_client import AsyncQdrantClient  # type: ignore

            self._client = AsyncQdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_s,
            )
            logger.info("✓ Qdrant initialized: %s", self.config.url)
        except Exception as e:
            logger.error("Qdrant init failed: %s", str(e), exc_info=True)
            raise

    async def upsert(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if self._client is None:
            raise RuntimeError("QdrantProvider not initialized")

        from qdrant_client.models import PointStruct  # type: ignore

        collection = collection_name or self.config.default_collection
        points = []
        for i, _id in enumerate(ids):
            points.append(
                PointStruct(
                    id=_id,
                    vector=vectors[i],
                    payload=(payloads[i] if payloads else {}) or {},
                )
            )

        await self._client.upsert(collection_name=collection, points=points)

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("QdrantProvider not initialized")

        collection = collection_name or self.config.default_collection

        results = await self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=int(limit),
            score_threshold=float(score_threshold),
        )

        out: List[Dict[str, Any]] = []
        for r in results:
            out.append({"id": str(r.id), "score": float(r.score), "payload": r.payload or {}})
        return out

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.close()
        self._client = None
        logger.info("QdrantProvider shutdown complete")


DEFAULT_CONFIG = QdrantConfig(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
    default_collection=os.getenv("QDRANT_COLLECTION", "documents"),
    timeout_s=_env_int("VECTORDB_TIMEOUT", 5),
)

default_provider = QdrantProvider(config=DEFAULT_CONFIG)

__all__ = ["QdrantConfig", "QdrantProvider", "DEFAULT_CONFIG", "default_provider"]
