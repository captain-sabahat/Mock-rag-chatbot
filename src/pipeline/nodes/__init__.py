"""
================================================================================
NODES PACKAGE - Individual Pipeline Processing Nodes
================================================================================

PURPOSE:
--------
Define individual pipeline nodes for document processing.

Nodes:
  1. IngestionNode - Parse file content (PDF, TXT, JSON, etc.)
  2. PreprocessingNode - Clean and normalize text
  3. ChunkingNode - Split text into chunks
  4. EmbeddingNode - Generate vector embeddings
  5. VectorDBNode - Store vectors in vector database

Each node:
  - Takes PipelineState as input
  - Performs specific processing
  - Updates PipelineState
  - Returns updated state
  - Handles errors gracefully

ARCHITECTURE:
--------------
     orchestrator._execute_node("ingestion")
              ↓
     ingestion_node(state) → state
              ↓
     orchestrator._execute_node("preprocessing")
              ↓
     preprocessing_node(state) → state
              ↓
     ... (continues)

USAGE:
------
    from src.pipeline.nodes import (
        ingestion_node,
        preprocessing_node,
        chunking_node,
        embedding_node,
        vectordb_node
    )
    
    state = await ingestion_node(state)
    state = await preprocessing_node(state)

================================================================================
"""

from .ingestion_node import ingestion_node
from .preprocessing_node import preprocessing_node
from .chunking_node import chunking_node
from .embedding_node import embedding_node
from .vectordb_node import vectordb_node

__all__ = [
    "ingestion_node",
    "preprocessing_node",
    "chunking_node",
    "embedding_node",
    "vectordb_node",
]
