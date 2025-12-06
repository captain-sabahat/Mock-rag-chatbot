"""
================================================================================
BASE VECTOR DATABASE MODULE
src/tools/vectordb/base_vectordb.py

MODULE PURPOSE:
───────────────
Abstract base class defining interface for vector databases.
Supports multiple vector store implementations.

WORKING & METHODOLOGY:
──────────────────────
1. VECTOR STORE OPERATIONS:
   - Index: Add vectors with metadata
   - Search: Find similar vectors (KNN)
   - Delete: Remove vectors by ID
   - Update: Modify vector data
   - Get: Retrieve vector by ID

2. CORE OPERATIONS:
   - index(): Add vectors to store
   - search(): Perform similarity search
   - batch_search(): Multiple queries
   - get_stats(): Store statistics
   - health_check(): Verify readiness

3. OPTIMIZATION:
   - Batch indexing
   - Efficient search (KNN)
   - Memory management
   - Metadata support

HOW IT CONTRIBUTES TO RAG PIPELINE:
──────────────────────────────────
- Store embedding vectors persistently
- Enable fast semantic search
- Support top-K retrieval
- Power ranking and filtering

VECTOR STORE PROPERTIES:
────────────────────────
- ID: Unique vector identifier
- Vector: Dense embedding (384-1024 dims)
- Metadata: Document attribution
- Similarity: Score for ranking

================================================================================
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time


@dataclass
class VectorStoreConfig:
    """Vector store configuration dataclass."""
    store_type: str
    dimension: int
    metric: str = "cosine"
    max_vectors: int = 1000000
    batch_size: int = 1000
    index_type: str = "flat"
    metadata: Dict[str, Any] = None


@dataclass
class SearchResult:
    """Search result dataclass."""
    id: str
    similarity: float
    vector: List[float]
    metadata: Dict[str, Any]
    rank: int


class BaseVectorDB(ABC):
    """
    Abstract base class for vector database implementations.
    
    Defines interface for:
    - Vector indexing
    - Similarity search
    - Metadata management
    - Store operations
    
    Example:
        >>> class MyVectorDB(BaseVectorDB):
        >>>     async def index(self, vectors, metadata):
        >>>         # Implementation
        >>>         pass
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base vector database.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters (config-driven, not hardcoded)
        self.store_type = config.get("store_type", "flat") if config else "flat"
        self.dimension = config.get("dimension", 384) if config else 384
        self.metric = config.get("metric", "cosine") if config else "cosine"
        self.max_vectors = config.get("max_vectors", 1000000) if config else 1000000
        self.batch_size = config.get("batch_size", 1000) if config else 1000
        self.index_type = config.get("index_type", "flat") if config else "flat"
        
        # Statistics
        self.total_vectors = 0
        self.total_searches = 0
        self.search_operations = 0
        self.total_search_time_ms = 0
        
        self.logger.info(
            "BaseVectorDB initialized (type=%s, dim=%d, metric=%s)" %
            (self.store_type, self.dimension, self.metric)
        )
    
    @abstractmethod
    async def index(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Index vectors in store.
        
        Args:
            vectors: List of embedding vectors
            ids: Unique IDs for vectors
            metadata: Optional metadata per vector
            
        Returns:
            Indexing result dictionary
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    async def batch_search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 5,
        filters: Optional[List[Dict[str, Any]]] = None
    ) -> List[List[SearchResult]]:
        """
        Search for multiple query vectors.
        
        Args:
            query_vectors: List of query embeddings
            top_k: Number of results per query
            filters: Optional filters per query
            
        Returns:
            List of result lists
        """
        results = []
        for i, query_vector in enumerate(query_vectors):
            filter_dict = filters[i] if filters and i < len(filters) else None
            result = await self.search(query_vector, top_k, filter_dict)
            results.append(result)
        
        return results
    
    async def _record_search(self, search_time_ms: float, result_count: int):
        """Record search operation."""
        self.total_searches += 1
        self.search_operations += 1
        self.total_search_time_ms += search_time_ms
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Statistics dictionary
        """
        avg_search_time = (self.total_search_time_ms / self.search_operations
                          if self.search_operations > 0 else 0)
        
        return {
            "total_vectors": self.total_vectors,
            "total_searches": self.total_searches,
            "operations": self.search_operations,
            "avg_search_time_ms": avg_search_time,
            "store_type": self.store_type,
            "dimension": self.dimension,
            "metric": self.metric,
        }
    
    async def health_check(self) -> bool:
        """
        Check vector database health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to search for test vector
            import numpy as np
            test_vector = np.random.randn(self.dimension).tolist()
            
            results = await self.search(test_vector, top_k=1)
            
            if results is not None:
                return True
            return False
        except Exception as e:
            self.logger.error("Health check failed: %s" % str(e))
            return False
    
    def cosine_similarity(
        self,
        vector1: List[float],
        vector2: List[float]
    ) -> float:
        """
        Compute cosine similarity between vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Similarity score (0-1)
        """
        import numpy as np
        arr1 = np.array(vector1, dtype=np.float32)
        arr2 = np.array(vector2, dtype=np.float32)
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, 0, 1))
    
    def euclidean_distance(
        self,
        vector1: List[float],
        vector2: List[float]
    ) -> float:
        """
        Compute euclidean distance between vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Distance value
        """
        import numpy as np
        arr1 = np.array(vector1, dtype=np.float32)
        arr2 = np.array(vector2, dtype=np.float32)
        
        distance = np.linalg.norm(arr1 - arr2)
        return float(distance)
    
    async def get_vector(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        Retrieve vector by ID.
        
        Args:
            vector_id: Vector identifier
            
        Returns:
            Tuple of (vector, metadata) or None
        """
        pass
    
    async def delete(self, vector_ids: List[str]) -> Dict[str, Any]:
        """
        Delete vectors by IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            Deletion result
        """
        pass
    
    async def clear(self) -> bool:
        """
        Clear all vectors from store.
        
        Returns:
            True if successful, False otherwise
        """
        pass
