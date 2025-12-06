"""
================================================================================
EMBEDDING NODE - Generate Vector Embeddings
================================================================================

PURPOSE:
--------
Fourth node in pipeline. Generates vector embeddings for text chunks.

Supported models:
  - OpenAI (text-embedding-3-small, text-embedding-3-large)
  - BGE (bge-small, bge-base)
  - Cohere
  - Local (sentence-transformers)

Responsibilities:
  - Generate embeddings for each chunk
  - Validate embedding dimensions
  - Handle batching for efficiency
  - Error recovery
  - Cost tracking (for API models)

INPUT:
------
  PipelineState.chunks = List[str] (text chunks)
  PipelineState.embedding_model = str (model name)
  PipelineState.embedding_dimension = int (expected dimension)

OUTPUT:
-------
  PipelineState.embeddings = List[List[float]] (vectors)
  PipelineState.checkpoints["embedding"] = updated

================================================================================
"""

import logging
from datetime import datetime
from typing import List

from src.pipeline.schemas import PipelineState, NodeStatus
from src.core import ValidationError, EmbeddingError

logger = logging.getLogger(__name__)


async def embedding_node(state: PipelineState) -> PipelineState:
    """
    Generate embeddings for chunks.
    
    Args:
        state: Pipeline state with chunks
        
    Returns:
        Updated state with embeddings
    """
    start_time = datetime.utcnow()
    
    logger.info(
        f"🧠 Embedding: Using {state.embedding_model} "
        f"({state.embedding_dimension}d)"
    )
    
    try:
        # Validate input
        if not state.chunks:
            raise ValidationError("No chunks to embed")
        
        # Generate embeddings
        if state.embedding_model == "openai":
            embeddings = await _embed_openai(state.chunks)
        elif state.embedding_model == "bge":
            embeddings = await _embed_bge(state.chunks)
        elif state.embedding_model == "local":
            embeddings = await _embed_local(state.chunks)
        else:
            raise ValidationError(f"Unknown embedding model: {state.embedding_model}")
        
        # Validate embeddings
        if len(embeddings) != len(state.chunks):
            raise ValidationError(
                f"Embedding count mismatch: {len(embeddings)} vs {len(state.chunks)}"
            )
        
        # Validate dimensions
        for i, emb in enumerate(embeddings):
            if len(emb) != state.embedding_dimension:
                raise ValidationError(
                    f"Embedding {i} has wrong dimension: "
                    f"{len(emb)} vs {state.embedding_dimension}"
                )
        
        # Update state
        state.embeddings = embeddings
        
        state.add_message(
            f"✅ Embedding: Generated {len(embeddings)} embeddings "
            f"({state.embedding_dimension}d)"
        )
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        state.update_checkpoint(
            "embedding",
            status=NodeStatus.COMPLETED,
            output_ready=True,
            output_data={
                "embedding_count": len(embeddings),
                "dimension": state.embedding_dimension,
                "model": state.embedding_model
            },
            duration_ms=duration_ms
        )
        
        logger.info(f"✅ Embedding complete: {len(embeddings)} vectors generated")
        return state
    
    except ValidationError as e:
        logger.error(f"❌ Embedding validation error: {e}")
        state.status = "error"
        state.add_error(f"Embedding validation: {e.message}")
        state.update_checkpoint(
            "embedding",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=e.message
        )
        return state
    
    except EmbeddingError as e:
        logger.error(f"❌ Embedding error: {e}")
        state.status = "error"
        state.add_error(f"Embedding error: {e.message}")
        state.update_checkpoint(
            "embedding",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=e.message
        )
        return state
    
    except Exception as e:
        logger.error(f"❌ Embedding failed: {str(e)}")
        state.status = "error"
        state.add_error(f"Embedding error: {str(e)}")
        state.update_checkpoint(
            "embedding",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=str(e)
        )
        return state


async def _embed_openai(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI API."""
    # Placeholder - implement with your OpenAI client
    logger.warning("⚠️  OpenAI embeddings not fully implemented")
    
    # Mock implementation for testing
    embeddings = []
    for chunk in chunks:
        # Generate mock embedding (1536 dimensions for text-embedding-3-small)
        embedding = [hash(chunk + str(i)) % 100 / 100.0 for i in range(1536)]
        embeddings.append(embedding)
    
    return embeddings


async def _embed_bge(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings using BGE model."""
    logger.warning("⚠️  BGE embeddings not fully implemented")
    
    # Mock implementation
    embeddings = []
    for chunk in chunks:
        embedding = [hash(chunk + str(i)) % 100 / 100.0 for i in range(384)]
        embeddings.append(embedding)
    
    return embeddings


async def _embed_local(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings using local sentence-transformers model."""
    logger.warning("⚠️  Local embeddings not fully implemented")
    
    # Mock implementation
    embeddings = []
    for chunk in chunks:
        embedding = [hash(chunk + str(i)) % 100 / 100.0 for i in range(768)]
        embeddings.append(embedding)
    
    return embeddings
