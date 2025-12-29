"""
FILE: src/providers/vectordb/faiss.py

FAISS VectorDB provider (local).

Env vars (from your screenshot):
- FAISS_INDEX_DIR=./data/faiss
- FAISS_TOP_K=5
- FAISS_SCORE_THRESHOLD=0.7

Design:
- One FAISS index per collection:
    {index_dir}/{collection_name}.index
    {index_dir}/{collection_name}.meta.jsonl
- meta.jsonl stores: {"id": "...", "payload": {...}}
- Uses cosine similarity via IndexFlatIP over normalized vectors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base import IVectorDBProvider

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
class FaissConfig:
    index_dir: str
    top_k_default: int
    score_threshold_default: float


class _FaissCollection:
    """
    In-memory representation of a FAISS collection.
    """

    def __init__(self, dim: int, index, ids: List[str], payloads: List[Dict[str, Any]]):
        self.dim = dim
        self.index = index
        self.ids = ids
        self.payloads = payloads


class FAISSProvider(IVectorDBProvider):
    """
    FAISS provider with simple persistence.

    Notes:
    - Requires embeddings dimension to match FAISS index dimension.
    - Normalizes vectors for cosine similarity using dot product (IP).
    """

    def __init__(self, config: FaissConfig):
        self.config = config
        self._collections: Dict[str, _FaissCollection] = {}
        self._lock = asyncio.Lock()
        logger.info(
            "FAISSProvider created: index_dir=%s top_k=%s threshold=%s",
            self.config.index_dir,
            self.config.top_k_default,
            self.config.score_threshold_default,
        )

    async def initialize(self) -> None:
        os.makedirs(self.config.index_dir, exist_ok=True)
        logger.info("✓ FAISSProvider initialized (dir=%s)", self.config.index_dir)

    def _paths(self, collection_name: str) -> Tuple[str, str]:
        index_path = os.path.join(self.config.index_dir, f"{collection_name}.index")
        meta_path = os.path.join(self.config.index_dir, f"{collection_name}.meta.jsonl")
        return index_path, meta_path

    def _normalize(self, vectors: List[List[float]]) -> List[List[float]]:
        # local, dependency-free normalize (avoid numpy requirement)
        out: List[List[float]] = []
        for v in vectors:
            norm = sum(x * x for x in v) ** 0.5
            if norm == 0:
                out.append(list(v))
            else:
                out.append([x / norm for x in v])
        return out

    def _load_collection_sync(self, collection_name: str, dim: int) -> _FaissCollection:
        import faiss  # type: ignore

        index_path, meta_path = self._paths(collection_name)

        # If index exists, load it; else create new
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            # Use inner product index; cosine = inner product of unit vectors
            index = faiss.IndexFlatIP(dim)

        ids: List[str] = []
        payloads: List[Dict[str, Any]] = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    ids.append(str(obj.get("id")))
                    payloads.append(obj.get("payload") or {})

        return _FaissCollection(dim=dim, index=index, ids=ids, payloads=payloads)

    async def _get_or_create_collection(self, collection_name: str, dim: int) -> _FaissCollection:
        async with self._lock:
            col = self._collections.get(collection_name)
            if col is not None:
                if col.dim != dim:
                    raise ValueError(
                        f"FAISS collection '{collection_name}' dim mismatch: "
                        f"existing={col.dim}, requested={dim}"
                    )
                return col

            col = await asyncio.to_thread(self._load_collection_sync, collection_name, dim)
            if col.dim != dim:
                # for loaded index, dim might be from index itself; keep strict
                raise ValueError(
                    f"FAISS collection '{collection_name}' dim mismatch after load: "
                    f"loaded={col.dim}, requested={dim}"
                )

            self._collections[collection_name] = col
            logger.info("Loaded/created FAISS collection: %s (dim=%s)", collection_name, dim)
            return col

    async def upsert(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not ids or not vectors:
            return
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        if payloads is not None and len(payloads) != len(ids):
            raise ValueError("payloads length mismatch")

        dim = len(vectors[0])
        for v in vectors:
            if len(v) != dim:
                raise ValueError("inconsistent vector dimensions in batch")

        col = await self._get_or_create_collection(collection_name, dim)

        # FAISS IndexFlatIP does not support in-place update by id without extra mapping;
        # this implementation appends and stores meta order = index row.
        vectors_n = self._normalize(vectors)

        def _add_and_persist():
            import faiss  # type: ignore
            import numpy as np  # type: ignore

            xb = np.array(vectors_n, dtype="float32")
            col.index.add(xb)

            # Append metadata
            for i, _id in enumerate(ids):
                col.ids.append(str(_id))
                col.payloads.append((payloads[i] if payloads else {}) or {})

            index_path, meta_path = self._paths(collection_name)
            faiss.write_index(col.index, index_path)

            # Rewrite meta file (simple, consistent)
            with open(meta_path, "w", encoding="utf-8") as f:
                for _id, pl in zip(col.ids, col.payloads):
                    f.write(json.dumps({"id": _id, "payload": pl}, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_add_and_persist)
        logger.info("✓ FAISS upsert complete: collection=%s items=%s", collection_name, len(ids))

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        if not query_vector:
            return []

        dim = len(query_vector)
        col = await self._get_or_create_collection(collection_name, dim)

        limit = int(limit or self.config.top_k_default)
        score_threshold = float(score_threshold if score_threshold is not None else self.config.score_threshold_default)

        qn = self._normalize([query_vector])[0]

        def _search():
            import numpy as np  # type: ignore

            q = np.array([qn], dtype="float32")
            scores, idxs = col.index.search(q, limit)

            results: List[Dict[str, Any]] = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0:
                    continue
                if float(score) < score_threshold:
                    continue

                _id = col.ids[idx] if idx < len(col.ids) else str(idx)
                payload = col.payloads[idx] if idx < len(col.payloads) else {}
                results.append({"id": _id, "score": float(score), "payload": payload})

            return results

        return await asyncio.to_thread(_search)

    async def shutdown(self) -> None:
        # Everything is persisted during upsert; just clear memory
        self._collections.clear()
        logger.info("FAISSProvider shutdown complete")


# =============================================================================
# Layer-2 defaults (from env) + exported instance (DEFAULT TOOL)
# =============================================================================

DEFAULT_CONFIG = FaissConfig(
    index_dir=os.getenv("FAISS_INDEX_DIR", "./data/faiss"),
    top_k_default=_env_int("FAISS_TOP_K", 5),
    score_threshold_default=_env_float("FAISS_SCORE_THRESHOLD", 0.7),
)

default_provider = FAISSProvider(config=DEFAULT_CONFIG)

__all__ = ["FaissConfig", "FAISSProvider", "DEFAULT_CONFIG", "default_provider"]
