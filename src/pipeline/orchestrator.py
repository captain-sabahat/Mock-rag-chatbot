"""
================================================================================
PIPELINE ORCHESTRATOR - v2.5 MINIMAL FIX
================================================================================

v2.5 CHANGES (MINIMAL - Only fixes needed):
✅ Fixed: NodeStatus instantiation now always includes ALL required fields
✅ Fixed: Proper error handling when creating NodeStatus with missing data
✅ Preserved: ALL existing orchestration logic, circuit breaker, progress mapping
✅ Preserved: Monitoring file writing, aggregation, state management

KEY FIX:
--------
When catching exceptions and creating NodeStatus, ALWAYS provide:
- node_name
- status
- request_id
- timestamp
- input_received
- input_valid
(All others have defaults now in schemas.py v2.5)

================================================================================
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.pipeline.schemas import (
    PipelineState, NodeStatus, NodeStatusEnum,
    PipelineStatus, CircuitBreakerState
)
from src.pipeline.nodes import (
    ingestion_node, preprocessing_node, chunking_node,
    embedding_node, vectordb_node
)
from src.core.exceptions import ValidationError, ProcessingError
from src.core import CircuitBreakerManager, CircuitBreakerConfig
from src.utils.monitoring_writer import (
    write_node_status,
    write_pipeline_status,
    write_circuit_breaker_state,
    ensure_request_dir,
)

logger = logging.getLogger(__name__)

# ============================================================================
# PROGRESS MAP
# ============================================================================

NODE_PROGRESS_MAP = {
    "ingestion": 20,
    "preprocessing": 40,
    "chunking": 60,
    "embedding": 80,
    "vectordb": 100,
}

# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

async def _load_pipeline_config() -> Dict[str, Any]:
    """Load pipeline configuration from config/defaults/*.yaml"""
    config = {}
    config_files = {
        "ingestion": "config/defaults/ingestion.yaml",
        "preprocessing": "config/defaults/preprocessing.yaml",
        "chunking": "config/defaults/chunking.yaml",
        "embedding": "config/defaults/embeddings.yaml",
        "vectordb": "config/defaults/vectordb.yaml",
    }

    for key, filepath in config_files.items():
        path = Path(filepath)
        if path.exists():
            try:
                import yaml
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
                    config[key] = data.get(key.lower(), {})
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {filepath}: {e}")
                config[key] = {}
        else:
            logger.warning(f"⚠️ Config file not found: {filepath}")
            config[key] = {}
    
    return config

# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

class PipelineOrchestrator:
    """Orchestrate RAG pipeline with monitoring and circuit breaker (v2.5)."""

    def __init__(self):
        """Initialize orchestrator."""
        self.circuit_breaker_manager = CircuitBreakerManager(CircuitBreakerConfig())
        self.node_order = [
            "ingestion",
            "preprocessing",
            "chunking",
            "embedding",
            "vectordb",
        ]
        logger.info("🚀 Pipeline orchestrator initialized (v2.5)")

    async def process_document(
        self,
        request_id: str,
        file_name: str,
        file_content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Process a document through the entire pipeline.
        
        Args:
            request_id: Unique session ID
            file_name: Original file name
            file_content: Raw file bytes
            metadata: Optional custom metadata
        
        Returns:
            request_id for status tracking
        
        Raises:
            ProcessingError: If pipeline execution fails
        """
        try:
            logger.info(
                f"📄 Processing: {file_name} "
                f"(request_id: {request_id})"
            )

            # Ensure monitoring directory exists
            ensure_request_dir(request_id)

            # Load config from YAML
            pipeline_config = await _load_pipeline_config()

            # Create initial state
            state = PipelineState(
                request_id=request_id,
                file_name=file_name,
                raw_content=file_content,
                metadata=metadata or {},
                started_at=datetime.utcnow(),
                ingestion_config=pipeline_config.get("ingestion", {}),
                preprocessing_config=pipeline_config.get("preprocessing", {}),
                chunking_config=pipeline_config.get("chunking", {}),
                embedding_config=pipeline_config.get("embedding", {}),
                vectordb_config=pipeline_config.get("vectordb", {}),
                chunking_strategy=pipeline_config.get("chunking", {}).get("strategy", "recursive"),
                chunk_size=pipeline_config.get("chunking", {}).get("chunk_size", 512),
                chunk_overlap=pipeline_config.get("chunking", {}).get("overlap", 50),
                embedding_model=pipeline_config.get("embedding", {}).get("model", "bge"),
                embedding_dimension=pipeline_config.get("embedding", {}).get("dimension", 768),
                vectordb_backend=pipeline_config.get("vectordb", {}).get("backend", "qdrant"),
            )

            state.add_message(f"🔄 Pipeline started for {file_name}")

            # ===== EXECUTE PIPELINE =====
            all_node_statuses: List[NodeStatus] = []
            circuit_breaker_opened = False

            for node_name in self.node_order:
                logger.info(f"\n{'='*80}")
                logger.info(f"🔄 Executing node: {node_name}")
                logger.info(f"{'='*80}")

                state.current_node = node_name

                # ✅ Derive progress from NODE_PROGRESS_MAP
                mapped_progress = NODE_PROGRESS_MAP.get(node_name)
                if mapped_progress is not None:
                    state.progress_percent = mapped_progress
                    logger.info(f"📊 Progress: {mapped_progress}%")

                # Execute node
                try:
                    if node_name == "ingestion":
                        state = await ingestion_node(state)
                    elif node_name == "preprocessing":
                        state = await preprocessing_node(state)
                    elif node_name == "chunking":
                        state = await chunking_node(state)
                    elif node_name == "embedding":
                        state = await embedding_node(state)
                    elif node_name == "vectordb":
                        state = await vectordb_node(state)

                except Exception as e:
                    logger.error(f"❌ Node {node_name} raised exception: {str(e)}")

                    # ✅ v2.5 FIX: Create error status with ALL required fields
                    node_status = NodeStatus(
                        node_name=node_name,
                        status=NodeStatusEnum.FAILED,
                        request_id=request_id,
                        timestamp=datetime.utcnow(),
                        input_received=False,
                        input_valid=False,
                        exception_type=type(e).__name__,
                        exception_message=str(e),
                        exception_severity=(
                            "CRITICAL"
                            if isinstance(e, (ValidationError, ProcessingError))
                            else "WARNING"
                        ),
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                    )
                    all_node_statuses.append(node_status)
                    state.node_statuses[node_name] = node_status

                else:
                    # Get node status if provided by node
                    if node_name in state.node_statuses:
                        node_status = state.node_statuses[node_name]
                        all_node_statuses.append(node_status)
                        logger.info(f"✅ Node status: {node_status.status.value}")
                    else:
                        # Fallback: create a completed status
                        node_status = NodeStatus(
                            node_name=node_name,
                            status=NodeStatusEnum.COMPLETED,
                            request_id=request_id,
                            timestamp=datetime.utcnow(),
                            input_received=True,
                            input_valid=True,
                            output_generated=True,
                            output_valid=True,
                        )
                        all_node_statuses.append(node_status)
                        state.node_statuses[node_name] = node_status
                        logger.info("✅ Node status: completed (implicit)")

                # WRITE MONITORING FILE
                node_debug_info = _extract_node_debug_info(state, node_name)
                node_status_data = node_status.to_dict()
                node_status_data["debug_info"] = node_debug_info

                await write_node_status(
                    request_id=request_id,
                    node_name=node_name,
                    status_data=node_status_data,
                )

                # Check if node failed
                if node_status.status == NodeStatusEnum.FAILED:
                    logger.error(
                        f"❌ Node failed: {node_status.exception_type} - "
                        f"{node_status.exception_message}"
                    )

                    # Check circuit breaker conditions
                    should_break, break_reason = await _check_circuit_breaker_conditions(
                        all_node_statuses
                    )

                    if should_break:
                        logger.error(f"🔴 Circuit breaker triggered: {break_reason}")
                        circuit_breaker_opened = True
                        state.status = "failed"
                        break
                    else:
                        logger.warning("⚠️ Node failed but circuit breaker not triggered")
                        break

            # ===== AGGREGATE FINAL STATUS =====
            logger.info(f"\n{'='*80}")
            logger.info("📊 Aggregating final pipeline status")
            logger.info(f"{'='*80}")

            pipeline_status = await _aggregate_pipeline_status(
                request_id,
                all_node_statuses,
                circuit_breaker_opened=circuit_breaker_opened,
                node_order=self.node_order,
            )

            # ✅ KEEP OLD LOGIC: 100% only if completed
            state.status = pipeline_status.status
            state.progress_percent = (
                100 if pipeline_status.status == "completed"
                else state.progress_percent
            )
            state.completed_at = datetime.utcnow()

            # WRITE FINAL MONITORING FILE
            await write_pipeline_status(
                request_id=request_id,
                pipeline_status=pipeline_status.to_dict(),
            )

            # WRITE CIRCUIT BREAKER STATE
            await write_circuit_breaker_state(
                self.circuit_breaker_manager.get_all_status()
            )

            logger.info(
                f"🎯 Pipeline {request_id} final status: "
                f"{pipeline_status.status} | "
                f"Circuit breaker: {pipeline_status.circuit_breaker_state.value}"
            )

            return request_id

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}", exc_info=True)
            raise ProcessingError(f"Pipeline execution failed: {str(e)}")

    async def get_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get current pipeline status from monitoring file.
        
        Reads: data/monitoring/nodes/{request_id}/pipeline_status.json
        """
        try:
            status_file = Path(f"./data/monitoring/nodes/{request_id}/pipeline_status.json")
            if not status_file.exists():
                return {
                    "request_id": request_id,
                    "status": "not_found",
                    "message": "Pipeline status file not found",
                }

            with open(status_file, 'r') as f:
                status_data = json.load(f)
            return status_data

        except Exception as e:
            logger.error(f"❌ Failed to get status: {str(e)}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
            }

    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "circuit_breakers": self.circuit_breaker_manager.get_all_status(),
            "nodes_configured": len(self.node_order),
        }

# ============================================================================
# HELPER: Extract node-enriched context from state
# ============================================================================

def _extract_node_debug_info(state: PipelineState, node_name: str) -> Dict[str, Any]:
    """
    Extract debug/context info from state enrichments.
    Nodes populate state fields; orchestrator reads them.
    """
    debug_info = {
        "node_name": node_name,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Nodes enrich state with these fields; orchestrator reads them
    if node_name == "preprocessing":
        debug_info["item_count"] = state.preprocess_item_count
        debug_info["sample_head"] = state.preprocess_head
        debug_info["sample_tail"] = state.preprocess_tail

    elif node_name == "chunking":
        debug_info["chunk_count"] = state.num_chunks
        debug_info["chunk_size_range"] = {
            "min": state.chunk_size_min,
            "max": state.chunk_size_max,
        }

    elif node_name == "embedding":
        debug_info["embedding_count"] = state.num_embeddings
        debug_info["embedding_samples"] = state.embedding_samples

    elif node_name == "vectordb":
        debug_info["batch_total"] = state.vectordb_batches_total
        debug_info["batch_done"] = state.vectordb_batches_done
        debug_info["upsert_count"] = state.vectordb_upsert_count

    return debug_info

# ============================================================================
# CIRCUIT BREAKER CONDITIONS (A-D)
# ============================================================================

async def _check_circuit_breaker_conditions(
    node_statuses: List[NodeStatus],
) -> tuple[bool, Optional[str]]:
    """
    Check all circuit breaker conditions sequentially.
    Returns: (should_break, reason)
    
    CONDITIONS:
    A: CRITICAL exception → trigger
    B: No output or invalid output → trigger
    C: Data transfer failure between nodes → trigger
    D: Time gap > 30s between nodes → trigger
    """
    logger.info("🔍 Checking circuit breaker conditions...")

    # ===== CONDITION A: CRITICAL EXCEPTION =====
    for node_status in node_statuses:
        if node_status.exception_severity == "CRITICAL":
            reason = (
                "Condition A (CRITICAL exception): "
                f"Node '{node_status.node_name}' raised CRITICAL exception: "
                f"{node_status.exception_type}"
            )
            logger.error(f"🔴 {reason}")
            return True, reason

    # ===== CONDITION B: NO OUTPUT OR INVALID OUTPUT =====
    for node_status in node_statuses:
        if not getattr(node_status, "output_generated", True):
            reason = (
                "Condition B (no output): "
                f"Node '{node_status.node_name}' failed to generate output"
            )
            logger.error(f"🔴 {reason}")
            return True, reason

        if not getattr(node_status, "output_valid", True):
            reason = (
                "Condition B (invalid output): "
                f"Node '{node_status.node_name}' generated invalid output"
            )
            logger.error(f"🔴 {reason}")
            return True, reason

    # ===== CONDITION C: DATA TRANSFER FAILURE =====
    for i in range(len(node_statuses) - 1):
        current_node = node_statuses[i]
        next_node = node_statuses[i + 1]

        if not getattr(current_node, "output_valid", True):
            reason = (
                "Condition C (transfer failure): "
                f"'{current_node.node_name}' output invalid, "
                "next node cannot proceed"
            )
            logger.error(f"🔴 {reason}")
            return True, reason

        if (not getattr(next_node, "input_received", True) or
            not getattr(next_node, "input_valid", True)):
            reason = (
                "Condition C (transfer failure): "
                f"'{current_node.node_name}' → '{next_node.node_name}' failed"
            )
            logger.error(f"🔴 {reason}")
            return True, reason

    # ===== CONDITION D: TIME GAP > 30S =====
    for i in range(len(node_statuses) - 1):
        current_node = node_statuses[i]
        next_node = node_statuses[i + 1]

        if current_node.end_time and next_node.start_time:
            time_gap = (next_node.start_time - current_node.end_time).total_seconds()
            if time_gap > 30.0:
                reason = (
                    "Condition D (time gap): "
                    f"Gap between '{current_node.node_name}' and '{next_node.node_name}': "
                    f"{time_gap:.2f}s > 30s threshold"
                )
                logger.error(f"🔴 {reason}")
                return True, reason

    logger.info("✅ All circuit breaker conditions passed")
    return False, None

# ============================================================================
# PIPELINE STATUS AGGREGATION
# ============================================================================

async def _aggregate_pipeline_status(
    request_id: str,
    node_statuses: List[NodeStatus],
    circuit_breaker_opened: bool = False,
    node_order: Optional[List[str]] = None,
) -> PipelineStatus:
    """
    Aggregate all node statuses into final pipeline status.
    
    Logic:
    1. If ALL nodes = COMPLETED → pipeline = COMPLETED, CB = CLOSED
    2. If ANY node = FAILED → Check circuit breaker conditions
    3. If CB triggered → pipeline = FAILED, CB = OPEN
    4. Build failure_summary with completed/pending nodes + CB reason
    """
    logger.info(f"📊 Aggregating status for {request_id}")

    if node_order is None:
        node_order = [
            "ingestion", "preprocessing", "chunking", "embedding", "vectordb"
        ]

    completed_count = sum(1 for s in node_statuses if s.status == NodeStatusEnum.COMPLETED)
    failed_count = sum(1 for s in node_statuses if s.status == NodeStatusEnum.FAILED)

    pipeline_status = PipelineStatus(
        request_id=request_id,
        status="completed",  # Default optimistic
        circuit_breaker_state=CircuitBreakerState.CLOSED,
        circuit_breaker_reason=None,
        node_statuses=node_statuses,
        timestamp=datetime.utcnow(),
    )

    # All nodes succeeded
    if completed_count == len(node_statuses):
        logger.info(f"🎉 All {len(node_statuses)} nodes COMPLETED")
        return pipeline_status

    # Some nodes failed - check circuit breaker
    if failed_count > 0:
        logger.warning(f"⚠️ {failed_count} nodes failed")

        # Find first failed node
        failed_node_status = next(
            (s for s in node_statuses if s.status == NodeStatusEnum.FAILED),
            None
        )
        failed_idx = (
            node_order.index(failed_node_status.node_name)
            if failed_node_status else -1
        )
        completed_nodes = [s.node_name for s in node_statuses if s.status == NodeStatusEnum.COMPLETED]
        pending_nodes = node_order[failed_idx + 1:] if failed_idx >= 0 else []

        # Build failure summary
        pipeline_status.failure_summary = {
            "failed_node": failed_node_status.node_name if failed_node_status else "unknown",
            "failure_reason": (
                f"{failed_node_status.exception_type}: {failed_node_status.exception_message}"
                if failed_node_status else "Unknown failure"
            ),
            "failure_severity": getattr(failed_node_status, "exception_severity", "WARNING"),
            "completed_nodes": completed_nodes,
            "pending_nodes": pending_nodes,
            "progress_completed": len(completed_nodes),
            "total_nodes": len(node_order),
            "cb_break_reason": None,
        }

        # If circuit breaker opened, ALWAYS set to failed
        if circuit_breaker_opened:
            pipeline_status.status = "failed"
            pipeline_status.circuit_breaker_state = CircuitBreakerState.OPEN
            pipeline_status.circuit_breaker_reason = "Circuit breaker opened - pipeline halted"
            pipeline_status.failure_summary["cb_break_reason"] = pipeline_status.circuit_breaker_reason
            logger.error("🔴 CIRCUIT BREAKER OPEN → Status forced to FAILED (v2.5)")
        else:
            # Backup: re-evaluate conditions
            should_break, break_reason = await _check_circuit_breaker_conditions(node_statuses)

            if should_break:
                pipeline_status.status = "failed"
                pipeline_status.circuit_breaker_state = CircuitBreakerState.OPEN
                pipeline_status.circuit_breaker_reason = break_reason
                pipeline_status.failure_summary["cb_break_reason"] = break_reason
                logger.error(f"🔴 Circuit breaker OPEN: {break_reason}")
            else:
                pipeline_status.status = "failed"
                pipeline_status.circuit_breaker_state = CircuitBreakerState.CLOSED
                pipeline_status.circuit_breaker_reason = "Non-critical node failure"
                pipeline_status.failure_summary["cb_break_reason"] = "Non-critical failure"
                logger.warning("⚠️ Node failed but CB not triggered")

        # Log failure summary
        if pipeline_status.failure_summary:
            summary = pipeline_status.failure_summary
            logger.error(f"""
================================================================================
📊 PIPELINE FAILURE SUMMARY
================================================================================
✅ Completed: {summary['progress_completed']}/{summary['total_nodes']} nodes
{chr(10).join(f" • {n}" for n in summary['completed_nodes']) if summary['completed_nodes'] else " (none)"}

❌ Failed: 1 node
• {summary['failed_node']} ({summary['failure_severity']})
Reason: {summary['failure_reason']}

⏳ Pending: {len(summary['pending_nodes'])} node(s)
{chr(10).join(f" • {n}" for n in summary['pending_nodes']) if summary['pending_nodes'] else " (none)"}

🔴 Circuit Breaker: {pipeline_status.circuit_breaker_state.value}
Reason: {summary['cb_break_reason'] or 'N/A'}

================================================================================
""")

    return pipeline_status

# ============================================================================
# SINGLETON PATTERN
# ============================================================================

_orchestrator: Optional[PipelineOrchestrator] = None

def get_orchestrator() -> PipelineOrchestrator:
    """Get or create singleton orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
        logger.info("✅ Orchestrator singleton created")
    return _orchestrator

def reset_orchestrator() -> None:
    """Reset singleton (for testing)."""
    global _orchestrator
    _orchestrator = None
    logger.info("🔄 Orchestrator singleton reset")
