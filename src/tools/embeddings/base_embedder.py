"""
================================================================================
BASE EMBEDDER MODULE
src/tools/embeddings/base_embedder.py

MODULE PURPOSE:
───────────────
Abstract base class defining interface for text embeddings.
Supports multiple embedding models and strategies.

WORKING & METHODOLOGY:
──────────────────────
1. EMBEDDING MODELS SUPPORTED:
   - Dense Retrieval (BGE, DPR, etc.)
   - Sparse Retrieval (BM25-style)
   - Hybrid models
   - Custom embedders

2. CORE OPERATIONS:
   - embed(): Convert text to vector
   - embed_batch(): Process multiple texts
   - embed_with_metadata(): Include document context
   - get_config(): Retrieve embedder configuration

3. OPTIMIZATION:
   - Batch processing
   - GPU acceleration ready
   - Caching support
   - Memory efficiency

HOW IT CONTRIBUTES TO RAG PIPELINE:
──────────────────────────────────
- Convert chunked text to semantic vectors
- Enable similarity search
- Support vector database indexing
- Power retrieval ranking

EMBEDDING PROPERTIES:
─────────────────────
- Vector: Dense embedding representation
- Dimension: Fixed embedding dimension (384-1024)
- Normalized: Optional L2 normalization
- Metadata: Document attribution

================================================================================
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
import numpy as np


@dataclass
class EmbeddingConfig:
    """Embedding configuration dataclass."""
    model_name: str
    dimension: int
    normalized: bool = True
    batch_size: int = 32
    max_retries: int = 3
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = None


@dataclass
class EmbeddingResult:
    """Embedding result dataclass."""
    text: str
    vector: List[float]
    dimension: int
    model_name: str
    normalized: bool
    processing_time_ms: float
    metadata: Dict[str, Any] = None


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding implementations.
    
    Defines interface for:
    - Text to vector conversion
    - Batch processing
    - Configuration management
    - Metadata handling
    
    Example:
        >>> class MyEmbedder(BaseEmbedder):
        >>>     async def embed(self, text):
        >>>         # Implementation
        >>>         pass
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base embedder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters (config-driven, not hardcoded)
        self.model_name = config.get("model_name", "default") if config else "default"
        self.dimension = config.get("dimension", 384) if config else 384
        self.normalized = config.get("normalized", True) if config else True
        self.batch_size = config.get("batch_size", 32) if config else 32
        self.max_retries = config.get("max_retries", 3) if config else 3
        self.timeout_seconds = config.get("timeout_seconds", 30) if config else 30
        
        # Statistics
        self.total_embeddings = 0
        self.total_texts = 0
        self.embedding_operations = 0
        self.total_processing_time_ms = 0
        
        self.logger.info(
            "BaseEmbedder initialized (model=%s, dim=%d, normalized=%s)" %
            (self.model_name, self.dimension, self.normalized)
        )
    
    @abstractmethod
    async def embed(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Embed single text.
        
        Args:
            text: Input text
            metadata: Optional document metadata
            
        Returns:
            EmbeddingResult with vector and metadata
        """
        pass
    
    async def embed_batch(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[EmbeddingResult]:
        """
        Embed multiple texts in batch.
        
        Args:
            texts: List of texts
            metadata: Optional metadata per text
            
        Returns:
            List of EmbeddingResults
        """
        results = []
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else None
            result = await self.embed(text, meta)
            results.append(result)
        
        return results
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        L2 normalize a vector.
        
        Args:
            vector: Input vector
            
        Returns:
            L2 normalized vector
        """
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        
        if norm == 0:
            return vector.tolist()
        
        return (arr / norm).tolist()
    
    def _validate_embedding(self, embedding: EmbeddingResult) -> bool:
        """
        Validate embedding properties.
        
        Args:
            embedding: Embedding to validate
            
        Returns:
            True if valid, False otherwise
        """
        if len(embedding.vector) != self.dimension:
            self.logger.warning(
                "Embedding dimension mismatch: %d (expected %d)" %
                (len(embedding.vector), self.dimension)
            )
            return False
        
        if not isinstance(embedding.vector, (list, np.ndarray)):
            self.logger.warning("Embedding vector must be list or ndarray")
            return False
        
        return True
    
    async def _record_embedding(self, embeddings: List[EmbeddingResult]):
        """Record embedding operation."""
        self.total_embeddings += len(embeddings)
        self.total_texts += len(embeddings)
        self.embedding_operations += 1
        self.total_processing_time_ms += sum(e.processing_time_ms for e in embeddings)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding statistics.
        
        Returns:
            Statistics dictionary
        """
        avg_processing_time = (self.total_processing_time_ms / self.embedding_operations
                              if self.embedding_operations > 0 else 0)
        
        return {
            "total_embeddings": self.total_embeddings,
            "total_texts": self.total_texts,
            "operations": self.embedding_operations,
            "avg_processing_time_ms": avg_processing_time,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "normalized": self.normalized,
        }
    
    async def health_check(self) -> bool:
        """
        Check embedder health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            test_text = "This is a test embedding for health check."
            result = await self.embed(test_text)
            
            if result and len(result.vector) == self.dimension:
                return True
            return False
        except Exception as e:
            self.logger.error("Health check failed: %s" % str(e))
            return False
    
    def similarity(
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
        arr1 = np.array(vector1, dtype=np.float32)
        arr2 = np.array(vector2, dtype=np.float32)
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, 0, 1))
    
    def batch_similarity(
        self,
        vectors: List[List[float]],
        query_vector: List[float]
    ) -> List[float]:
        """
        Compute similarities between multiple vectors and query.
        
        Args:
            vectors: List of vectors
            query_vector: Query vector
            
        Returns:
            List of similarity scores
        """
        similarities = []
        for vector in vectors:
            sim = self.similarity(vector, query_vector)
            similarities.append(sim)
        
        return similarities