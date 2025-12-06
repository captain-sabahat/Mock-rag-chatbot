"""
================================================================================
PIPELINE PACKAGE - LangGraph-based Orchestration & Nodes
================================================================================

PURPOSE:
--------
Orchestrate the RAG document processing pipeline using LangGraph.

Pipeline Flow:
  Ingestion → Preprocessing → Chunking → Embedding → VectorDB

This package provides:
  - Orchestrator: Manages pipeline execution
  - Schemas: State and node definitions
  - Nodes: Individual processing nodes
  - LangGraph integration: Stateful workflow

ARCHITECTURE:
--------------
     API Route
           ↓
     orchestrator.py (execute_pipeline)
           ↓
     LangGraph
           ↓
     nodes/ (ingestion, preprocessing, chunking, embedding, vectordb)
           ↓
     tools/ (chunker, embedder, vectordb client)
     cache/ (session store)
           ↓
     Result

USAGE EXAMPLE:
--------------
    from src.pipeline import get_orchestrator
    
    orchestrator = get_orchestrator()
    result = await orchestrator.process_document(
        request_id="req_123",
        file_name="document.pdf",
        file_content=bytes(...)
    )

================================================================================
"""

from .orchestrator import (
    get_orchestrator,
    PipelineOrchestrator,
)

from .schemas import (
    PipelineState,
    NodeInput,
    NodeOutput,
    NodeStatus,
    NodeCheckpointData,
)

__all__ = [
    "get_orchestrator",
    "PipelineOrchestrator",
    "PipelineState",
    "NodeInput",
    "NodeOutput",
    "NodeStatus",
    "NodeCheckpointData"
]
