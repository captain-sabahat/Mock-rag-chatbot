"""
================================================================================
PIPELINE PACKAGE - Orchestrator, Schemas, and Nodes
================================================================================

PURPOSE:
--------
Expose the high-level pipeline orchestrator, data schemas, and node package
through a clean import surface.

Modules:
- schemas       : Pydantic models for NodeStatus, PipelineStatus, CircuitBreakerState, etc.
- orchestrator  : PipelineOrchestrator with circuit breaker + monitoring integration
- nodes         : Individual processing nodes (ingestion, preprocessing, chunking,
                  embedding, vectordb) using the A-B-C pattern and writing node statuses.

TYPICAL USAGE:
--------------
from src.pipeline import (
    PipelineOrchestrator,
    get_orchestrator,
    PipelineState,
    PipelineStatus,
    NodeStatus,
)

# Get singleton orchestrator
orchestrator = get_orchestrator()

# Process a document
request_id = await orchestrator.process_document(
    request_id="req_123",
    file_name="doc.pdf",
    file_content=b"...",
    metadata={"source": "upload"},
)

# Check status (from monitoring JSON written by orchestrator)
status = await orchestrator.get_status(request_id)
print(status)

ARCHITECTURE OVERVIEW:
----------------------
user_request
    ↓
orchestrator.process_document()
    ├─ Load YAML config (ingestion, preprocessing, chunking, embedding, vectordb)
    ├─ Build PipelineState
    ├─ Execute nodes in order:
    │    ingestion  → preprocessing → chunking → embedding → vectordb
    │    (each node updates state.node_statuses[node_name])
    ├─ After each node:
    │    - NodeStatus recorded
    │    - Orchestrator writes per-node JSON via monitoring writer
    │      data/monitoring/nodes/{request_id}/{node}_node.json
    ├─ Aggregate all NodeStatus into PipelineStatus
    │    - Apply circuit breaker conditions A–D
    │    - Set circuit_breaker_state and reason
    └─ Write aggregated pipeline_status.json (single source of truth)
         data/monitoring/nodes/{request_id}/pipeline_status.json

API INTEGRATION:
----------------
- /api/ingest/upload         → triggers orchestrator.process_document()
- /api/monitor/status/{id}   → reads pipeline_status.json + per-node files
- /api/monitor/health        → exposes circuit breaker state + pipeline metrics
- /api/monitor/metrics       → per-node and per-request metrics

================================================================================
"""

from .schemas import (
    PipelineState,
    NodeStatus,
    NodeStatusEnum,
    PipelineStatus,
    CircuitBreakerState,
)

from .orchestrator import (
    PipelineOrchestrator,
    get_orchestrator,
    reset_orchestrator,
)

from . import nodes  # package with ingestion_node, preprocessing_node, etc.

__all__ = [
    # Schemas
    "PipelineState",
    "NodeStatus",
    "NodeStatusEnum",
    "PipelineStatus",
    "CircuitBreakerState",
    # Orchestrator
    "PipelineOrchestrator",
    "get_orchestrator",
    "reset_orchestrator",
    # Nodes package
    "nodes",
]
