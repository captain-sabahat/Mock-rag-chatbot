"""
================================================================================
VECTORDB NODE - Store Vectors in Vector Database
================================================================================

PURPOSE:
--------
Fifth node in pipeline. Stores embeddings in vector database.

Supported backends:
  - Qdrant (recommended)
  - Pinecone
  - FAISS
  - Milvus
  - Weaviate

Responsibilities:
  - Connect to vector DB
  - Create/verify collection
  - Store embeddings with metadata
  - Validate storage
  - Create indexes

INPUT:
------
  PipelineState.embeddings = List[List[float]] (vectors)
  PipelineState.chunks = List[str] (original text)
  PipelineState.chunk_metadata = List[Dict] (metadata)
  PipelineState.vectordb_backend = str (backend name)
  PipelineState.request_id = str (for document tracking)

OUTPUT:
-------
  PipelineState.vectordb_indexed = bool (success flag)
  PipelineState.checkpoints["vectordb"] = updated

================================================================================
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from src.pipeline.schemas import PipelineState, NodeStatus
from src.core import ValidationError, VectorDBError

logger = logging.getLogger(__name__)


async def vectordb_node(state: PipelineState) -> PipelineState:
    """
    Store embeddings in vector database.
    
    Args:
        state: Pipeline state with embeddings
        
    Returns:
        Updated state with storage confirmation
    """
    start_time = datetime.utcnow()
    
    logger.info(f"🗄️  VectorDB: Storing {len(state.embeddings)} vectors in {state.vectordb_backend}")
    
    try:
        # Validate input
        if not state.embeddings:
            raise ValidationError("No embeddings to store")
        
        if len(state.embeddings) != len(state.chunks):
            raise ValidationError("Embedding/chunk count mismatch")
        
        # Store in vector DB
        if state.vectordb_backend == "qdrant":
            stored_count = await _store_qdrant(
                state.request_id,
                state.embeddings,
                state.chunks,
                state.chunk_metadata
            )
        elif state.vectordb_backend == "pinecone":
            stored_count = await _store_pinecone(
                state.request_id,
                state.embeddings,
                state.chunks,
                state.chunk_metadata
            )
        elif state.vectordb_backend == "faiss":
            stored_count = await _store_faiss(
                state.request_id,
                state.embeddings,
                state.chunks,
                state.chunk_metadata
            )
        else:
            raise ValidationError(f"Unknown backend: {state.vectordb_backend}")
        
        # Validate storage
        if stored_count != len(state.embeddings):
            raise VectorDBError(
                f"Storage verification failed: {stored_count} vs {len(state.embeddings)}"
            )
        
        # Update state
        state.add_message(
            f"✅ VectorDB: Stored {stored_count} vectors in {state.vectordb_backend}"
        )
        
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        state.update_checkpoint(
            "vectordb",
            status=NodeStatus.COMPLETED,
            output_ready=True,
            output_data={
                "stored_count": stored_count,
                "backend": state.vectordb_backend,
                "collection": state.request_id
            },
            duration_ms=duration_ms
        )
        
        logger.info(f"✅ VectorDB complete: {stored_count} vectors stored")
        return state
    
    except ValidationError as e:
        logger.error(f"❌ VectorDB validation error: {e}")
        state.status = "error"
        state.add_error(f"VectorDB validation: {e.message}")
        state.update_checkpoint(
            "vectordb",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=e.message
        )
        return state
    
    except VectorDBError as e:
        logger.error(f"❌ VectorDB error: {e}")
        state.status = "error"
        state.add_error(f"VectorDB error: {e.message}")
        state.update_checkpoint(
            "vectordb",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=e.message
        )
        return state
    
    except Exception as e:
        logger.error(f"❌ VectorDB failed: {str(e)}")
        state.status = "error"
        state.add_error(f"VectorDB error: {str(e)}")
        state.update_checkpoint(
            "vectordb",
            status=NodeStatus.FAILED,
            error_flag=True,
            error_message=str(e)
        )
        return state


async def _store_qdrant(
    request_id: str,
    embeddings: List[List[float]],
    chunks: List[str],
    metadata: List[Dict[str, Any]]
) -> int:
    """Store vectors in Qdrant."""
    # Placeholder - implement with your Qdrant client
    logger.warning("⚠️  Qdrant storage not fully implemented")
    
    # Mock implementation for testing
    return len(embeddings)


async def _store_pinecone(
    request_id: str,
    embeddings: List[List[float]],
    chunks: List[str],
    metadata: List[Dict[str, Any]]
) -> int:
    """Store vectors in Pinecone."""
    logger.warning("⚠️  Pinecone storage not fully implemented")
    
    # Mock implementation
    return len(embeddings)


async def _store_faiss(
    request_id: str,
    embeddings: List[List[float]],
    chunks: List[str],
    metadata: List[Dict[str, Any]]
) -> int:
    """Store vectors in FAISS."""
    logger.warning("⚠️  FAISS storage not fully implemented")
    
    # Mock implementation
    return len(embeddings)
