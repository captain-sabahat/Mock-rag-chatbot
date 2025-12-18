"""
================================================================================
PIPELINE SCHEMAS - Pydantic Models for Pipeline State & Monitoring (FIXED v2.5)
================================================================================

FIXES IN v2.5:
- NodeStatus now has ALL required fields properly defaulted
- PipelineState adds node-enrichment fields for monitoring (safe, optional)
- Removed per-node status flags that broke Pydantic immutability
- All fields are explicitly typed and validated

PURPOSE:
--------
Define Pydantic models for type-safe pipeline state management with
circuit breaker and node status monitoring integration.

================================================================================
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

# ============================================================================
# ENUMS - Status and State Definitions
# ============================================================================

class NodeStatusEnum(str, Enum):
    """Node execution status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CIRCUIT_OPEN = "circuit_open"

class ExceptionSeverity(str, Enum):
    """Exception severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    UNKNOWN = "UNKNOWN"

# ============================================================================
# CORE NODE STATUS SCHEMA (v2.5 - All required fields have defaults)
# ============================================================================

class NodeStatus(BaseModel):
    """
    Status report from a single node execution (v2.5 FIXED).
    
    ✅ ALL required fields have proper defaults or are optional
    ✅ Captures ABC pattern: input validation (A), processing (B), output validation (C)
    ✅ Written immediately to monitoring file after node completion
    
    DO NOT try to instantiate without providing all required fields manually.
    """

    # Node identification
    node_name: str = Field(..., description="Node name")
    status: NodeStatusEnum = Field(default=NodeStatusEnum.PENDING)
    request_id: str = Field(default="", description="Pipeline execution ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # ======== A METHOD: INPUT VALIDATION ========
    input_received: bool = Field(default=False)
    input_valid: bool = Field(default=False)

    # ======== B METHOD: PROCESSING ========
    # (Execution happens in node logic, only result tracked here)

    # ======== C METHOD: OUTPUT VALIDATION ========
    output_generated: bool = Field(default=False)
    output_valid: bool = Field(default=False)

    # Exception tracking
    exception_type: Optional[str] = Field(default=None)
    exception_message: Optional[str] = Field(default=None)
    exception_severity: Optional[str] = Field(default=None)

    # Timing
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    execution_time_ms: Optional[float] = Field(default=None)

    # Circuit breaker
    circuit_breaker_triggered: bool = Field(default=False)
    circuit_breaker_reason: Optional[str] = Field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for monitoring file."""
        return {
            "node_name": self.node_name,
            "status": self.status.value,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "input": {
                "received": self.input_received,
                "valid": self.input_valid,
            },
            "output": {
                "generated": self.output_generated,
                "valid": self.output_valid,
            },
            "exception": {
                "type": self.exception_type,
                "message": self.exception_message,
                "severity": self.exception_severity,
            },
            "timing": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "execution_time_ms": self.execution_time_ms,
            },
            "circuit_breaker": {
                "triggered": self.circuit_breaker_triggered,
                "reason": self.circuit_breaker_reason,
            },
        }

# ============================================================================
# PIPELINE STATUS AGGREGATION (For Final Status)
# ============================================================================

class PipelineStatus(BaseModel):
    """
    Aggregated pipeline status across all nodes (v2.4).
    
    This is the SINGLE SOURCE OF TRUTH for dashboard/API.
    Updated AFTER all nodes have reported their status.
    """

    request_id: str
    status: str  # "completed", "failed", "partially_completed"
    circuit_breaker_state: CircuitBreakerState
    circuit_breaker_reason: Optional[str] = None
    node_statuses: List[NodeStatus] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pipeline_start_time: Optional[datetime] = None
    pipeline_end_time: Optional[datetime] = None
    total_execution_time_ms: Optional[float] = None
    failure_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for monitoring file."""
        return {
            "request_id": self.request_id,
            "status": self.status,
            "circuit_breaker": {
                "state": self.circuit_breaker_state.value,
                "reason": self.circuit_breaker_reason,
            },
            "nodes": [ns.to_dict() for ns in self.node_statuses],
            "timestamp": self.timestamp.isoformat(),
            "timing": {
                "pipeline_start": (
                    self.pipeline_start_time.isoformat()
                    if self.pipeline_start_time else None
                ),
                "pipeline_end": (
                    self.pipeline_end_time.isoformat()
                    if self.pipeline_end_time else None
                ),
                "total_execution_time_ms": self.total_execution_time_ms,
            },
            "failure_summary": self.failure_summary,
        }

# ============================================================================
# LEGACY SCHEMAS (Kept for backward compatibility)
# ============================================================================

class NodeCheckpointData(BaseModel):
    """Checkpoint data for a node (legacy)."""
    node_name: str = Field(..., description="Node name")
    status: NodeStatusEnum = Field(default=NodeStatusEnum.PENDING)
    input_valid: bool = Field(True)
    output_ready: bool = Field(False)
    error_flag: bool = Field(False)
    error_message: Optional[str] = Field(None)
    input_data: Optional[Dict[str, Any]] = Field(None)
    output_data: Optional[Dict[str, Any]] = Field(None)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: Optional[float] = Field(None)

class NodeInput(BaseModel):
    """Input to a pipeline node."""
    node_name: str = Field(..., description="Name of the node")
    request_id: str = Field(..., description="Session ID")
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NodeOutput(BaseModel):
    """Output from a pipeline node."""
    node_name: str = Field(..., description="Node that produced this")
    status: NodeStatusEnum = Field(..., description="Execution status")
    data: Dict[str, Any] = Field(...)
    error_message: Optional[str] = Field(None)
    processing_time_ms: float = Field(...)

# ============================================================================
# MAIN PIPELINE STATE (v2.5 - Added node-enrichment fields, removed flags)
# ============================================================================

class PipelineState(BaseModel):
    """
    Complete pipeline state (LangGraph state) - v2.5 FIXED.
    
    ✅ No per-node status flags (causes Pydantic immutability errors)
    ✅ Node-enrichment fields OPTIONAL and have safe defaults
    ✅ Config is loaded from YAML, NOT hardcoded
    ✅ Nodes ONLY update data fields, not flags
    
    PATTERN:
    - Orchestrator loads config once
    - Each node reads config from state.{node}_config
    - Each node enriches state with metrics (num_chunks, embedding_count, etc)
    - Orchestrator reads enriched state for monitoring
    - NodeStatus is written separately (not state flags)
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
    raw_content: Optional[bytes] = Field(None)
    parsed_text: Optional[str] = Field(None)
    cleaned_text: Optional[str] = Field(None)
    chunks: List[str] = Field(default_factory=list)
    embeddings: List[List[float]] = Field(default_factory=list)
    chunk_metadata: List[Dict[str, Any]] = Field(default_factory=list)

    # ========================================================================
    # NODE STATUS TRACKING (Per-node monitoring)
    # ========================================================================
    node_statuses: Dict[str, NodeStatus] = Field(default_factory=dict)
    checkpoints: Dict[str, NodeCheckpointData] = Field(default_factory=dict)

    # ========================================================================
    # CONFIGURATION (Loaded from config/defaults/*.yaml)
    # ========================================================================
    ingestion_config: Dict[str, Any] = Field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = Field(default_factory=dict)
    chunking_config: Dict[str, Any] = Field(default_factory=dict)
    chunking_strategy: str = Field("recursive")
    chunk_size: int = Field(512)
    chunk_overlap: int = Field(50)
    embedding_config: Dict[str, Any] = Field(default_factory=dict)
    embedding_model: str = Field("bge")
    embedding_dimension: int = Field(768)
    vectordb_config: Dict[str, Any] = Field(default_factory=dict)
    vectordb_backend: str = Field("qdrant")

    # ========================================================================
    # NODE-ENRICHMENT FIELDS (Optional metrics added by nodes during execution)
    # ========================================================================
    # Ingestion enrichment
    ingestion_size_bytes: Optional[int] = Field(None)
    ingestion_char_count: Optional[int] = Field(None)
    ingestion_word_count: Optional[int] = Field(None)
    ingestion_format_detected: Optional[str] = Field(None)
    ingestion_encoding: Optional[str] = Field(None)

    # Preprocessing enrichment
    preprocess_item_count: Optional[int] = Field(None)
    preprocess_head: Optional[str] = Field(None)
    preprocess_tail: Optional[str] = Field(None)

    # Chunking enrichment
    num_chunks: Optional[int] = Field(None)
    chunk_size_min: Optional[int] = Field(None)
    chunk_size_max: Optional[int] = Field(None)

    # Embedding enrichment
    num_embeddings: Optional[int] = Field(None)
    embedding_samples: Optional[List[List[float]]] = Field(None)

    # VectorDB enrichment
    vectordb_batches_total: Optional[int] = Field(None)
    vectordb_batches_done: Optional[int] = Field(None)
    vectordb_upsert_count: Optional[int] = Field(None)

    # ========================================================================
    # METADATA & TRACKING
    # ========================================================================
    metadata: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    messages: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    def add_message(self, message: str) -> None:
        """Add a message to pipeline log."""
        self.messages.append(message)

    def add_error(self, error: str) -> None:
        """Add an error to error list."""
        self.errors.append(error)

    def update_checkpoint(
        self,
        node_name: str,
        status: NodeStatusEnum = NodeStatusEnum.PENDING,
        input_valid: bool = True,
        output_ready: bool = False,
        error_flag: bool = False,
        error_message: Optional[str] = None,
        output_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None
    ) -> None:
        """Update node checkpoint (legacy compatibility)."""
        checkpoint = NodeCheckpointData(
            node_name=node_name,
            status=status,
            input_valid=input_valid,
            output_ready=output_ready,
            error_flag=error_flag,
            error_message=error_message,
            output_data=output_data,
            duration_ms=duration_ms,
        )
        self.checkpoints[node_name] = checkpoint

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_abc123",
                "file_name": "document.pdf",
                "status": "processing",
                "current_node": "ingestion",
                "progress_percent": 20,
            }
        }
