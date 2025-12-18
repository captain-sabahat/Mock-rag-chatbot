"""
================================================================================
BGE EMBEDDER IMPLEMENTATION
src/tools/embeddings/bge_embedder.py

PURPOSE:
- Concrete implementation of BaseEmbedder for BGE models
- Reads ALL config from injected config dict (no hardcoding)
- Provider-specific embedding logic only
- NO config loading (that's registry_embed.py's job)

KEY: This file only does BGE embedding. Config comes from BaseEmbedder init.
================================================================================
"""

import logging
import time
from typing import Dict, Any, List
from .embed_registry import BaseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)

class BGEEmbedder(BaseEmbedder):
    """
    BGE (BAAI General Embeddings) embedder implementation.
    
    Reads config from constructor, never hardcodes parameters.
    """
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        """
        Initialize BGE embedder.
        
        Args:
            provider_name: "huggingface" (passed by factory)
            config: Provider config dict from YAML
                   Contains: model_name, dimension, batch_size, etc.
        """
        super().__init__(provider_name, config)
        
        # BGE-specific initialization
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model_name = config.get("model_name", "BAAI/bge-base-en-v1.5")
            self.device = config.get("device", "cpu")
            
            self.logger.info(f"📦 Loading BGE model: {self.model_name}...")
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                #trust_remote_code=config.get("trust_remote_code", False), otherwise manual loading 
            )
            
            # Verify dimension
            test_embedding = self.model.encode(["test"])
            actual_dim = len(test_embedding[0])
            
            if actual_dim != self.dimension:
                self.logger.warning(
                    f"⚠️ Dimension mismatch: config={self.dimension}, "
                    f"actual={actual_dim}. Using actual."
                )
                self.dimension = actual_dim
            
            self.logger.info(
                f"✅ BGEEmbedder ready: model={self.model_name}, "
                f"dimension={self.dimension}, device={self.device}, "
                f"normalize={self.normalize_embeddings}"
            )
        
        except ImportError:
            self.logger.error(
                "❌ sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            self.logger.error(f"❌ BGE initialization failed: {str(e)}", exc_info=True)
            raise
    
    async def embed(self, text: str) -> EmbeddingResult:
        """
        Embed a single text using BGE.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with vector and metadata
        """
        start_time = time.time()
        
        try:
            # Encode with sentence-transformers
            # normalize_embeddings is built-in parameter
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.normalize_embeddings,
            )
            
            # Ensure correct dimension
            if len(embedding) != self.dimension:
                self.logger.warning(
                    f"⚠️ Embedding dimension mismatch: "
                    f"got {len(embedding)}, expected {self.dimension}"
                )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = EmbeddingResult(
                text=text,
                vector=embedding.tolist(),
                dimension=len(embedding),
                provider=self.provider_name,
                processing_time_ms=elapsed_ms,
                normalized=self.normalize_embeddings,
                metadata={
                    "model": self.model_name,
                    "text_length": len(text),
                    "tokens_approx": len(text) // 4,  # Rough estimate
                }
            )
            
            return result
        
        except Exception as e:
            self.logger.error(
                f"❌ Embedding failed for text (len={len(text)}): {str(e)}",
                exc_info=True
            )
            raise
    
    async def embed_batch(self, texts: List[str]) -> tuple:
        """
        Embed multiple texts efficiently using batch processing.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            (results list, total count)
        """
        start_time = time.time()
        
        try:
            self.logger.info(
                f"📦 Embedding batch: {len(texts)} texts, "
                f"batch_size={self.batch_size}..."
            )
            
            # Encode all at once (sentence-transformers handles batching)
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
            
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                result = EmbeddingResult(
                    text=text,
                    vector=embedding.tolist(),
                    dimension=len(embedding),
                    provider=self.provider_name,
                    processing_time_ms=(time.time() - start_time) * 1000 / len(texts),
                    normalized=self.normalize_embeddings,
                    metadata={
                        "model": self.model_name,
                        "batch_index": i,
                        "text_length": len(text),
                    }
                )
                results.append(result)
            
            total_time_ms = (time.time() - start_time) * 1000
            texts_per_sec = len(texts) / (total_time_ms / 1000)
            
            self.logger.info(
                f"✅ Batch complete: {len(results)} embeddings in "
                f"{total_time_ms:.1f}ms ({texts_per_sec:.1f} texts/sec)"
            )
            
            return results, len(results)
        
        except Exception as e:
            self.logger.error(
                f"❌ Batch embedding failed ({len(texts)} texts): {str(e)}",
                exc_info=True
            )
            raise

__all__ = ['BGEEmbedder']