"""
================================================================================
MOCK EMBEDDER MODULE
src/tools/embeddings/mock_embedder.py

MODULE PURPOSE:
───────────────
Mock embedder for testing and development.
Deterministic embeddings without model dependencies.

WORKING & METHODOLOGY:
──────────────────────
1. MOCK EMBEDDING STRATEGY:
   - Deterministic hash-based vectors
   - No model loading required
   - Fast processing
   - Reproducible results

2. TESTING BENEFITS:
   - No external dependencies
   - Fast unit tests
   - Deterministic behavior
   - Perfect for CI/CD

3. MOCK FEATURES:
   - Same interface as real embedders
   - Configurable dimension
   - Text-based hashing
   - Similarity computation

HOW IT CONTRIBUTES TO RAG PIPELINE:
──────────────────────────────────
- Enable testing without models
- Fast prototyping
- Unit test support
- Development/debugging aid

PERFORMANCE:
────────────
- Speed: < 0.1ms per text (fastest)
- Dimension: Configurable (384, 768, 1024)
- Memory: Minimal
- Dependencies: None

================================================================================
"""

from typing import Dict, Any, List, Optional
import logging
import time
import hashlib
import numpy as np
from .base_embedder import BaseEmbedder, EmbeddingResult


class MockEmbedder(BaseEmbedder):
    """
    Mock embedder for testing and development.
    
    Features:
    - No external dependencies
    - Deterministic embeddings
    - Fast processing
    - Same interface as real embedders
    
    Example:
        >>> embedder = MockEmbedder({
        ...     "dimension": 384,
        ...     "normalized": True
        ... })
        >>> result = await embedder.embed(text)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mock embedder.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Mock-specific config
        self.seed_offset = config.get("seed_offset", 42) if config else 42
        self.deterministic = config.get("deterministic", True) if config else True
        
        self.logger.info(
            "MockEmbedder initialized (dim=%d, deterministic=%s)" %
            (self.dimension, self.deterministic)
        )
    
    async def embed(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Generate mock embedding for text.
        
        Args:
            text: Input text
            metadata: Document metadata
            
        Returns:
            EmbeddingResult with mock vector
        """
        start_time = time.time()
        
        if not text or len(text) == 0:
            self.logger.warning("Empty text provided")
            return EmbeddingResult(
                text=text,
                vector=[0.0] * self.dimension,
                dimension=self.dimension,
                model_name=self.model_name,
                normalized=self.normalized,
                processing_time_ms=0,
                metadata=metadata,
            )
        
        try:
            # Generate deterministic vector from text hash
            vector = self._generate_mock_vector(text)
            
            # Normalize if configured
            if self.normalized:
                vector = self._normalize_vector(vector)
            
            # Calculate processing time (very fast for mock)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = EmbeddingResult(
                text=text,
                vector=vector,
                dimension=self.dimension,
                model_name=self.model_name,
                normalized=self.normalized,
                processing_time_ms=elapsed_ms,
                metadata={
                    **(metadata or {}),
                    "mock_embedder": True,
                    "deterministic": self.deterministic,
                }
            )
            
            # Validate
            if self._validate_embedding(result):
                await self._record_embedding([result])
                return result
            else:
                self.logger.error("Mock embedding validation failed")
                return result
                
        except Exception as e:
            self.logger.error("Mock embedding generation failed: %s" % str(e))
            raise
    
    def _generate_mock_vector(self, text: str) -> List[float]:
        """
        Generate deterministic mock vector from text hash.
        
        Args:
            text: Input text
            
        Returns:
            Mock embedding vector
        """
        # Hash text to seed
        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        seed = (hash_int + self.seed_offset) % (2**31)
        
        # Generate deterministic vector
        np.random.seed(seed)
        vector = np.random.randn(self.dimension).astype(np.float32)
        
        return vector.tolist()
    
    async def embed_batch(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[EmbeddingResult]:
        """
        Embed multiple texts (mock).
        
        Args:
            texts: List of texts
            metadata: Optional metadata per text
            
        Returns:
            List of mock EmbeddingResults
        """
        start_time = time.time()
        results = []
        
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else None
            result = await self.embed(text, meta)
            results.append(result)
        
        elapsed_ms = (time.time() - start_time) * 1000
        self.logger.info("Mock embedded batch of %d texts in %.2fms" % (len(texts), elapsed_ms))
        
        return results
    
    async def embed_with_variation(
        self,
        text: str,
        variation: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Embed text with variation (for testing).
        
        Args:
            text: Input text
            variation: Variation type (default, noisy, extreme)
            metadata: Document metadata
            
        Returns:
            EmbeddingResult
        """
        # Generate base vector
        vector = self._generate_mock_vector(text)
        
        # Apply variation
        arr = np.array(vector, dtype=np.float32)
        
        if variation == "noisy":
            # Add noise
            noise = np.random.randn(self.dimension) * 0.1
            arr = arr + noise
        elif variation == "extreme":
            # Extreme values
            arr = arr * 5
        
        # Normalize if needed
        if self.normalized:
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        
        elapsed_ms = 0.05  # Mock elapsed time
        
        result = EmbeddingResult(
            text=text,
            vector=arr.tolist(),
            dimension=self.dimension,
            model_name=self.model_name,
            normalized=self.normalized,
            processing_time_ms=elapsed_ms,
            metadata={
                **(metadata or {}),
                "mock_variation": variation,
                "mock_embedder": True,
            }
        )
        
        return result
    
    async def embed_stress_test(
        self,
        count: int = 1000
    ) -> Dict[str, Any]:
        """
        Run stress test embeddings.
        
        Args:
            count: Number of embeddings to generate
            
        Returns:
            Stress test results
        """
        start_time = time.time()
        
        texts = [f"Test text number {i}" for i in range(count)]
        results = await self.embed_batch(texts)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "count": count,
            "total_time_ms": elapsed_ms,
            "avg_time_per_embedding_ms": elapsed_ms / count,
            "embeddings_per_sec": count / (elapsed_ms / 1000),
            "success_count": len(results),
        }