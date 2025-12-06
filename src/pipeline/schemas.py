"""
================================================================================
PIPELINE SCHEMAS - Pydantic Models for Pipeline State
================================================================================

PURPOSE:
--------
Define Pydantic models for type-safe pipeline state management.

Schemas:
  - PipelineState: Complete pipeline state (LangGraph state)
  - NodeInput: Input to any pipeline node
  - NodeOutput: Output from any pipeline node

Benefits:
  ✅ Type-safe state passing
  ✅ Automatic validation
  ✅ Clear data flow
  ✅ JSON serialization ready
  ✅ OpenAPI documentation

ARCHITECTURE:
--------------
     Orchestrator creates PipelineState
           ↓
     Passes to each Node
           ↓
     Node validates NodeInput
           ↓
     Node executes
           ↓
     Node returns NodeOutput
           ↓
     Orchestrator updates PipelineState

================================================================================
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class NodeStatus(str, Enum):
    """Node execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeInput(BaseModel):
    """Input to a pipeline node."""
    
    node_name: str = Field(..., description="Name of the node")
    request_id: str = Field(..., description="Session ID")
    data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "node_name": "chunking",
                "request_id": "req_123",
                "data": {"text": "Large document text..."},
                "metadata": {"source": "pdf"}
            }
        }


class NodeOutput(BaseModel):
    """Output from a pipeline node."""
    
    node_name: str = Field(..., description="Node that produced this")
    status: NodeStatus = Field(..., description="Execution status")
    data: Dict[str, Any] = Field(..., description="Output data")
    error_message: Optional[str] = Field(None, description="Error if failed")
    processing_time_ms: float = Field(..., description="Execution time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "node_name": "chunking",
                "status": "completed",
                "data": {"chunks": ["chunk1", "chunk2"], "count": 2},
                "error_message": None,
                "processing_time_ms": 125.5
            }
        }


class NodeCheckpointData(BaseModel):
    """Checkpoint data for a node."""
    
    node_name: str = Field(..., description="Node name")
    status: NodeStatus = Field(default=NodeStatus.PENDING)
    input_valid: bool = Field(True, description="Input validation passed")
    output_ready: bool = Field(False, description="Output ready for next node")
    error_flag: bool = Field(False, description="Error occurred")
    error_message: Optional[str] = Field(None)
    input_data: Optional[Dict[str, Any]] = Field(None)
    output_data: Optional[Dict[str, Any]] = Field(None)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: Optional[float] = Field(None)


class PipelineState(BaseModel):
    """
    Complete pipeline state (LangGraph state).
    
    This is passed through all nodes and updated at each step.
    Represents the entire processing state from input to output.
    """
    
    # ========================================================================
    # SESSION IDENTIFICATION
    # ========================================================================
    
    request_id: str = Field(..., description="Unique session ID")
    file_name: str = Field(..., description="Original file name")
    
    # ========================================================================
    # CURRENT STATE
    # ========================================================================
    
    status: str = Field(
        "processing",
        description="Overall status (processing/completed/error)"
    )
    current_node: str = Field(
        "ingestion",
        description="Currently executing node"
    )
    progress_percent: int = Field(
        0,
        ge=0,
        le=100,
        description="Progress percentage"
    )
    
    # ========================================================================
    # DATA FLOW (Passed between nodes)
    # ========================================================================
    
    raw_content: Optional[bytes] = Field(
        None,
        description="Raw file content"
    )
    parsed_text: Optional[str] = Field(
        None,
        description="Parsed text from document"
    )
    cleaned_text: Optional[str] = Field(
        None,
        description="Cleaned/preprocessed text"
    )
    chunks: List[str] = Field(
        default_factory=list,
        description="Text chunks from chunking node"
    )
    embeddings: List[List[float]] = Field(
        default_factory=list,
        description="Vector embeddings from embedding node"
    )
    chunk_metadata: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Metadata for each chunk"
    )
    
    # ========================================================================
    # NODE CHECKPOINTS (Track each node's progress)
    # ========================================================================
    
    checkpoints: Dict[str, NodeCheckpointData] = Field(
        default_factory=dict,
        description="Per-node execution state"
    )
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Ingestion
    ingestion_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ingestion node config"
    )
    
    # Preprocessing
    preprocessing_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Preprocessing node config"
    )
    
    # Chunking
    chunking_strategy: str = Field(
        "recursive",
        description="Chunking strategy (recursive, token, sliding_window)"
    )
    chunk_size: int = Field(512, description="Chunk size")
    chunk_overlap: int = Field(50, description="Chunk overlap")
    
    # Embedding
    embedding_model: str = Field(
        "openai",
        description="Embedding model (openai, bge, cohere, etc.)"
    )
    embedding_dimension: int = Field(1536, description="Embedding vector dimension")
    
    # VectorDB
    vectordb_backend: str = Field(
        "qdrant",
        description="VectorDB backend (qdrant, pinecone, faiss, etc.)"
    )
    
    # ========================================================================
    # CIRCUIT BREAKER & OVERRIDE
    # ========================================================================
    
    circuit_break_triggered: bool = Field(
        False,
        description="Circuit breaker activated"
    )
    manual_override_flag: bool = Field(
        False,
        description="Manual override to bypass circuit breaker"
    )
    retry_count: int = Field(0, description="Current retry count")
    max_retries: int = Field(3, description="Max allowed retries")
    
    # ========================================================================
    # MESSAGES & LOGGING
    # ========================================================================
    
    messages: List[str] = Field(
        default_factory=list,
        description="Audit trail of events"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Errors encountered"
    )
    
    # ========================================================================
    # METADATA & TIMING
    # ========================================================================
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    started_at: Optional[datetime] = Field(None, description="Pipeline start time")
    completed_at: Optional[datetime] = Field(None, description="Pipeline completion time")
    total_duration_ms: float = Field(0.0, description="Total execution time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_123",
                "file_name": "document.pdf",
                "status": "processing",
                "current_node": "chunking",
                "progress_percent": 40,
                "chunk_size": 512,
                "embedding_model": "openai",
                "chunks": ["chunk1", "chunk2"],
                "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
                "messages": ["Parsed 50 pages", "Split into 200 chunks"]
            }
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def add_message(self, message: str) -> None:
        """Add a message to audit trail."""
        self.messages.append(f"[{datetime.utcnow().isoformat()}] {message}")

    def add_error(self, error: str) -> None:
        """Add an error to error list."""
        self.errors.append(f"[{datetime.utcnow().isoformat()}] {error}")

    def update_checkpoint(
        self,
        node_name: str,
        status: NodeStatus = NodeStatus.PENDING,
        **kwargs
    ) -> None:
        """Update checkpoint for a node."""
        if node_name not in self.checkpoints:
            self.checkpoints[node_name] = NodeCheckpointData(
                node_name=node_name,
                status=status
            )
        else:
            checkpoint = self.checkpoints[node_name]
            checkpoint.status = status
            
            for key, value in kwargs.items():
                if hasattr(checkpoint, key):
                    setattr(checkpoint, key, value)

    def get_checkpoint(self, node_name: str) -> Optional[NodeCheckpointData]:
        """Get checkpoint for a node."""
        return self.checkpoints.get(node_name)

    def is_node_completed(self, node_name: str) -> bool:
        """Check if node completed successfully."""
        checkpoint = self.get_checkpoint(node_name)
        return checkpoint and checkpoint.status == NodeStatus.COMPLETED
