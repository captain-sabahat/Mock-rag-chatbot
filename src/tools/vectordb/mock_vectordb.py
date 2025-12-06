"""
================================================================================
MOCK VECTOR DATABASE MODULE
src/tools/vectordb/mock_vectordb.py

MODULE PURPOSE:
───────────────
Mock vector database for testing and development.
Deterministic embeddings without external dependencies.

WORKING & METHODOLOGY:
──────────────────────
1. MOCK VECTOR STORE:
   - In-memory storage
   - No external dependencies
   - Deterministic search
   - Fast operations

2. TESTING FEATURES:
   - Same interface as real stores
   - Configurable behavior
   - Reproducible results
   - Perfect for CI/CD

3. MOCK OPERATIONS:
   - Index: Add vectors to memory
   - Search: Find similar (mock)
   - Delete: Remove vectors
   - Get: Retrieve by ID
   - Statistics: Track operations

HOW IT CONTRIBUTES TO RAG PIPELINE:
──────────────────────────────────
- Enable testing without FAISS
- Fast prototyping
- Unit test support
- Development/debugging aid

PERFORMANCE:
────────────
- Speed: <1ms per search
- Memory: In-memory only
- Dimension: Configurable
- Dependencies: None

================================================================================
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import numpy as np
from .base_vectordb import BaseVectorDB, SearchResult


class MockVectorDB(BaseVectorDB):
    """
    Mock vector database for testing.
    
    Features:
    - No external dependencies
    - Deterministic search
    - Fast processing
    - Same interface as real stores
    
    Example:
        >>> mock = MockVectorDB({
        ...     "dimension": 384
        ... })
        >>> await mock.index(vectors, ids, metadata)
        >>> results = await mock.search(query_vector)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mock vector database.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Mock-specific config
        self.deterministic = config.get("deterministic", True) if config else True
        self.search_delay_ms = config.get("search_delay_ms", 0) if config else 0
        
        # In-memory storage
        self.vectors = {}
        self.metadata_store = {}
        self.vector_list = []
        self.id_list = []
        
        self.logger.info(
            "MockVectorDB initialized (deterministic=%s)" % self.deterministic
        )
    
    async def index(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Index vectors (mock).
        
        Args:
            vectors: List of vectors
            ids: Vector IDs
            metadata: Optional metadata
            
        Returns:
            Indexing result
        """
        start_time = time.time()
        
        if not vectors:
            return {"status": "failed", "message": "Empty vectors"}
        
        try:
            for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
                self.vectors[vector_id] = np.array(vector, dtype=np.float32)
                self.vector_list.append(np.array(vector, dtype=np.float32))
                self.id_list.append(vector_id)
                
                if metadata and i < len(metadata):
                    self.metadata_store[vector_id] = metadata[i]
                else:
                    self.metadata_store[vector_id] = {}
            
            self.total_vectors += len(vectors)
            elapsed_ms = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "vectors_indexed": len(vectors),
                "total_vectors": self.total_vectors,
                "time_ms": elapsed_ms,
            }
        except Exception as e:
            self.logger.error("Indexing failed: %s" % str(e))
            return {"status": "failed", "message": str(e)}
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors (mock).
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        if not self.vector_list:
            return []
        
        try:
            query_arr = np.array(query_vector, dtype=np.float32)
            
            # Compute similarities
            similarities = []
            for i, stored_vector in enumerate(self.vector_list):
                similarity = self.cosine_similarity(
                    query_arr.tolist(),
                    stored_vector.tolist()
                )
                similarities.append({
                    "index": i,
                    "id": self.id_list[i],
                    "similarity": similarity,
                    "vector": stored_vector,
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Apply filters if needed
            filtered = similarities
            if filters:
                filtered = []
                for sim in similarities:
                    if self._matches_filters(
                        self.metadata_store[sim["id"]],
                        filters
                    ):
                        filtered.append(sim)
            
            # Get top-k
            top_results = filtered[:top_k]
            
            # Create results
            results = []
            for rank, result in enumerate(top_results):
                search_result = SearchResult(
                    id=result["id"],
                    similarity=float(result["similarity"]),
                    vector=result["vector"].tolist(),
                    metadata=self.metadata_store[result["id"]],
                    rank=rank + 1,
                )
                results.append(search_result)
            
            # Record search
            await self._record_search(
                (time.time() - start_time) * 1000,
                len(results)
            )
            
            return results
            
        except Exception as e:
            self.logger.error("Search failed: %s" % str(e))
            return []
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters."""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    async def get_vector(
        self,
        vector_id: str
    ) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get vector by ID."""
        if vector_id not in self.vectors:
            return None
        
        return (
            self.vectors[vector_id].tolist(),
            self.metadata_store.get(vector_id, {})
        )
    
    async def delete(self, vector_ids: List[str]) -> Dict[str, Any]:
        """Delete vectors by IDs."""
        deleted_count = 0
        for vector_id in vector_ids:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
                del self.metadata_store[vector_id]
                # Remove from lists
                self.vector_list = [v for i, v in enumerate(self.vector_list)
                                   if self.id_list[i] != vector_id]
                self.id_list = [id for id in self.id_list if id != vector_id]
                deleted_count += 1
        
        self.total_vectors -= deleted_count
        
        return {
            "status": "success",
            "deleted": deleted_count,
            "remaining": self.total_vectors,
        }
    
    async def clear(self) -> bool:
        """Clear all vectors."""
        self.vectors.clear()
        self.metadata_store.clear()
        self.vector_list.clear()
        self.id_list.clear()
        self.total_vectors = 0
        return True
    
    async def stress_test(
        self,
        vector_count: int = 1000,
        query_count: int = 100,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run stress test on mock store.
        
        Args:
            vector_count: Vectors to index
            query_count: Queries to run
            top_k: Results per query
            
        Returns:
            Stress test results
        """
        start_time = time.time()
        
        # Generate and index vectors
        vectors = [np.random.randn(self.dimension).tolist() for _ in range(vector_count)]
        ids = [f"vec_{i}" for i in range(vector_count)]
        
        await self.index(vectors, ids)
        
        # Run queries
        for _ in range(query_count):
            query = np.random.randn(self.dimension).tolist()
            await self.search(query, top_k=top_k)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "vectors_indexed": vector_count,
            "queries_run": query_count,
            "total_time_ms": elapsed_ms,
            "avg_time_per_query_ms": elapsed_ms / query_count,
            "queries_per_sec": query_count / (elapsed_ms / 1000),
        }
