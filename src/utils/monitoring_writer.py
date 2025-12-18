"""
================================================================================
MONITORING WRITER - Centralized Helper for Real-Time Pipeline Monitoring
================================================================================

PURPOSE:
--------
Provide unified, reusable monitoring functions for:
- Per-node JSON writing (A-B-C pattern tracking)
- Pipeline status aggregation
- Circuit breaker state persistence
- Real-time metric updates

Used by:
- Orchestrator (writes node + pipeline status)
- API routes (reads monitoring files + exposes via REST)
- Monitor dashboard (visualizes monitoring data)

ARCHITECTURE:
--------------
Write operations:
├─ write_node_status() - Write individual node A-B-C status
├─ write_pipeline_status() - Write aggregated pipeline status
├─ update_circuit_breaker_state() - Update CB state in monitoring
└─ aggregate_all_monitoring() - Merge all JSON files

Read operations:
├─ read_node_status() - Read individual node status
├─ read_pipeline_status() - Read aggregated pipeline status
├─ read_all_node_statuses() - Read all nodes for request
└─ read_circuit_breaker_state() - Read CB state

Directory structure:
data/monitoring/
├── nodes/
│   └── {request_id}/
│       ├── ingestion_node.json      (A-B-C pattern)
│       ├── preprocessing_node.json  (A-B-C pattern)
│       ├── chunking_node.json       (A-B-C pattern)
│       ├── embedding_node.json      (A-B-C pattern)
│       ├── vectordb_node.json       (A-B-C pattern)
│       └── pipeline_status.json     (AGGREGATED - SINGLE SOURCE OF TRUTH)
├── pipeline_logs.json               (Real-time logs)
└── circuit_breaker_state.json       (CB status across all services)

================================================================================
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

MONITORING_BASE_DIR = Path("./data/monitoring")
NODES_DIR = MONITORING_BASE_DIR / "nodes"

def ensure_monitoring_dirs() -> None:
    """Create all necessary monitoring directories."""
    MONITORING_BASE_DIR.mkdir(parents=True, exist_ok=True)
    NODES_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug(f"✅ Monitoring directories ready: {MONITORING_BASE_DIR}")

def ensure_request_dir(request_id: str) -> Path:
    """Create request-specific monitoring directory."""
    request_dir = NODES_DIR / request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir

# ============================================================================
# WRITE OPERATIONS - Per-Node Status
# ============================================================================

async def write_node_status(
    request_id: str,
    node_name: str,
    status_data: Dict[str, Any]
) -> Path:
    """
    Write individual node status file (A-B-C pattern tracking).
    
    File: data/monitoring/nodes/{request_id}/{node_name}_node.json
    
    Args:
        request_id: Pipeline request ID
        node_name: Node name (ingestion, preprocessing, chunking, etc.)
        status_data: Node status dict with A-B-C tracking
    
    Returns:
        Path to written file
    
    Example status_data:
    {
        "node_name": "ingestion",
        "status": "completed",
        "timestamp": "2025-12-13T20:30:45.123Z",
        
        # A-B-C Pattern
        "input_received": true,
        "input_valid": true,
        "execution_completed": true,
        "output_generated": true,
        "output_valid": true,
        
        # Metrics
        "execution_time_ms": 1234,
        "input_size_bytes": 10240,
        "output_size_bytes": 8192,
        
        # Errors
        "exception_type": null,
        "exception_message": null,
        "exception_severity": "INFO"
    }
    """
    try:
        request_dir = ensure_request_dir(request_id)
        node_file = request_dir / f"{node_name}_node.json"
        
        # Add timestamp if not present
        if "timestamp" not in status_data:
            status_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        with open(node_file, "w") as f:
            json.dump(status_data, f, indent=2, default=str)
        
        logger.info(f"📝 Node status written: {node_file}")
        return node_file
    
    except Exception as e:
        logger.error(f"❌ Failed to write node status: {str(e)}")
        raise

# ============================================================================
# WRITE OPERATIONS - Pipeline Status (Aggregated)
# ============================================================================

async def write_pipeline_status(
    request_id: str,
    pipeline_status: Dict[str, Any]
) -> Path:
    """
    Write aggregated pipeline status (SINGLE SOURCE OF TRUTH).
    
    File: data/monitoring/nodes/{request_id}/pipeline_status.json
    
    This is the definitive status file that aggregates all node statuses
    and circuit breaker state.
    
    Args:
        request_id: Pipeline request ID
        pipeline_status: Aggregated status dict
    
    Returns:
        Path to written file
    
    Example pipeline_status:
    {
        "request_id": "req_abc123",
        "status": "completed",
        "timestamp": "2025-12-13T20:30:45.123Z",
        
        "circuit_breaker": {
            "state": "CLOSED",
            "reason": null,
            "triggered_condition": null
        },
        
        "nodes": {
            "ingestion": {"status": "completed"},
            "preprocessing": {"status": "completed"},
            "chunking": {"status": "completed"},
            "embedding": {"status": "completed"},
            "vectordb": {"status": "completed"}
        },
        
        "metrics": {
            "total_execution_time_ms": 5000,
            "total_input_size_bytes": 10240,
            "total_output_size_bytes": 8192
        }
    }
    """
    try:
        request_dir = ensure_request_dir(request_id)
        status_file = request_dir / "pipeline_status.json"
        
        # Add timestamp if not present
        if "timestamp" not in pipeline_status:
            pipeline_status["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        with open(status_file, "w") as f:
            json.dump(pipeline_status, f, indent=2, default=str)
        
        logger.info(f"📊 Pipeline status written (SOURCE OF TRUTH): {status_file}")
        return status_file
    
    except Exception as e:
        logger.error(f"❌ Failed to write pipeline status: {str(e)}")
        raise

# ============================================================================
# WRITE OPERATIONS - Circuit Breaker State
# ============================================================================

async def write_circuit_breaker_state(
    cb_state: Dict[str, Any]
) -> Path:
    """
    Write circuit breaker state (global, not per-request).
    
    File: data/monitoring/circuit_breaker_state.json
    
    Args:
        cb_state: Circuit breaker state dict
    
    Returns:
        Path to written file
    
    Example cb_state:
    {
        "timestamp": "2025-12-13T20:30:45.123Z",
        "overall_state": "CLOSED",
        "breakers": {
            "ingestion": {
                "state": "CLOSED",
                "failure_count": 0,
                "success_count": 100,
                "last_failure": null
            },
            "embedding": {
                "state": "OPEN",
                "failure_count": 5,
                "last_failure": "2025-12-13T20:30:40.000Z"
            }
        }
    }
    """
    try:
        ensure_monitoring_dirs()
        cb_file = MONITORING_BASE_DIR / "circuit_breaker_state.json"
        
        if "timestamp" not in cb_state:
            cb_state["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        with open(cb_file, "w") as f:
            json.dump(cb_state, f, indent=2, default=str)
        
        logger.info(f"🔌 Circuit breaker state written: {cb_file}")
        return cb_file
    
    except Exception as e:
        logger.error(f"❌ Failed to write CB state: {str(e)}")
        raise

# ============================================================================
# READ OPERATIONS - Individual Files
# ============================================================================

def read_node_status(request_id: str, node_name: str) -> Optional[Dict[str, Any]]:
    """
    Read individual node status file.
    
    File: data/monitoring/nodes/{request_id}/{node_name}_node.json
    
    Returns:
        Dict with node status or None if file not found
    """
    try:
        node_file = NODES_DIR / request_id / f"{node_name}_node.json"
        
        if not node_file.exists():
            logger.warning(f"⚠️ Node status file not found: {node_file}")
            return None
        
        with open(node_file, "r") as f:
            return json.load(f)
    
    except Exception as e:
        logger.error(f"❌ Failed to read node status: {str(e)}")
        return None

def read_pipeline_status(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Read aggregated pipeline status (SINGLE SOURCE OF TRUTH).
    
    File: data/monitoring/nodes/{request_id}/pipeline_status.json
    
    Returns:
        Dict with pipeline status or None if file not found
    """
    try:
        status_file = NODES_DIR / request_id / "pipeline_status.json"
        
        if not status_file.exists():
            logger.warning(f"⚠️ Pipeline status file not found: {status_file}")
            return None
        
        with open(status_file, "r") as f:
            return json.load(f)
    
    except Exception as e:
        logger.error(f"❌ Failed to read pipeline status: {str(e)}")
        return None

def read_circuit_breaker_state() -> Optional[Dict[str, Any]]:
    """
    Read global circuit breaker state.
    
    File: data/monitoring/circuit_breaker_state.json
    
    Returns:
        Dict with CB state or None if file not found
    """
    try:
        cb_file = MONITORING_BASE_DIR / "circuit_breaker_state.json"
        
        if not cb_file.exists():
            logger.warning(f"⚠️ CB state file not found: {cb_file}")
            return None
        
        with open(cb_file, "r") as f:
            return json.load(f)
    
    except Exception as e:
        logger.error(f"❌ Failed to read CB state: {str(e)}")
        return None

# ============================================================================
# READ OPERATIONS - Aggregated
# ============================================================================

def read_all_node_statuses(request_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Read all per-node status files for a request.
    
    Reads: data/monitoring/nodes/{request_id}/*_node.json
    
    Returns:
        Dict mapping node_name → status_dict
    """
    try:
        request_dir = NODES_DIR / request_id
        node_statuses = {}
        
        if not request_dir.exists():
            logger.warning(f"⚠️ Request directory not found: {request_dir}")
            return node_statuses
        
        # Find all *_node.json files
        for node_file in request_dir.glob("*_node.json"):
            node_name = node_file.stem.replace("_node", "")
            
            try:
                with open(node_file, "r") as f:
                    node_statuses[node_name] = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Failed to read {node_file}: {str(e)}")
        
        logger.info(f"✅ Read {len(node_statuses)} node statuses for {request_id}")
        return node_statuses
    
    except Exception as e:
        logger.error(f"❌ Failed to read all node statuses: {str(e)}")
        return {}

def read_monitoring_response(request_id: str) -> Dict[str, Any]:
    """
    Read complete monitoring response for API.
    
    Aggregates:
    - pipeline_status.json (SINGLE SOURCE OF TRUTH)
    - All per-node status files
    - Circuit breaker state
    
    Returns:
        Complete monitoring dict ready for API response
    """
    try:
        pipeline_status = read_pipeline_status(request_id)
        node_statuses = read_all_node_statuses(request_id)
        cb_state = read_circuit_breaker_state()
        
        return {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "pipeline_status": pipeline_status or {},
            "per_node_statuses": node_statuses,
            "circuit_breaker": cb_state or {},
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to build monitoring response: {str(e)}")
        return {
            "request_id": request_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_monitoring_directory(request_id: str) -> Path:
    """Get path to monitoring directory for request."""
    return NODES_DIR / request_id

def get_monitoring_files(request_id: str) -> Dict[str, Path]:
    """Get all monitoring files for a request."""
    request_dir = get_monitoring_directory(request_id)
    
    return {
        "pipeline_status": request_dir / "pipeline_status.json",
        "ingestion_node": request_dir / "ingestion_node.json",
        "preprocessing_node": request_dir / "preprocessing_node.json",
        "chunking_node": request_dir / "chunking_node.json",
        "embedding_node": request_dir / "embedding_node.json",
        "vectordb_node": request_dir / "vectordb_node.json",
        "circuit_breaker": MONITORING_BASE_DIR / "circuit_breaker_state.json",
    }

def monitoring_files_exist(request_id: str) -> bool:
    """Check if monitoring files exist for request."""
    pipeline_status = NODES_DIR / request_id / "pipeline_status.json"
    return pipeline_status.exists()

def list_all_requests() -> List[str]:
    """List all request IDs with monitoring data."""
    try:
        if not NODES_DIR.exists():
            return []
        
        return [d.name for d in NODES_DIR.iterdir() if d.is_dir()]
    
    except Exception as e:
        logger.error(f"❌ Failed to list requests: {str(e)}")
        return []

def cleanup_old_monitoring(days: int = 7) -> int:
    """
    Delete monitoring data older than N days.
    
    Returns:
        Number of request directories deleted
    """
    try:
        from datetime import timedelta
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        deleted_count = 0
        
        if not NODES_DIR.exists():
            return 0
        
        for request_dir in NODES_DIR.iterdir():
            if not request_dir.is_dir():
                continue
            
            # Get directory modification time
            mtime = datetime.fromtimestamp(request_dir.stat().st_mtime)
            
            if mtime < cutoff_time:
                import shutil
                shutil.rmtree(request_dir)
                deleted_count += 1
                logger.info(f"🗑️ Deleted old monitoring: {request_dir}")
        
        logger.info(f"✅ Cleanup completed: {deleted_count} directories deleted")
        return deleted_count
    
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {str(e)}")
        return 0
