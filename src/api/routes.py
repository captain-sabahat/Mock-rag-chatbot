# ============================================================================
# API Routes - RAG Pipeline Endpoints (POST-Based Monitoring) v2.5 UPDATED
# ============================================================================

"""
API Routes - RAG Pipeline Endpoints with POST-based monitoring

v2.5 UPDATED WITH PROGRESS AGGREGATION:
- Routes call orchestrator
- Routes read monitoring/*.json files
- Progress NOW properly aggregated from node completions (20% per node)
- No route-node coupling enforced

FLOW:
1. /ingest/upload → call orchestrator.process_document() async
2. Orchestrator writes monitoring/nodes/{request_id}/{node}_node.json
3. progress_aggregation reads node statuses → calculates progress
4. /ingest/status → reads aggregated progress (NO hardcoded 10%)

PROGRESS MAPPING (5 nodes):
- 0%: Queued
- 20%: Ingestion complete
- 40%: Preprocessing complete
- 60%: Chunking complete
- 80%: Embedding complete
- 100%: VectorDB complete

✅ v2.5-3 FIX (STATUS CHECK):
- When final_cb_state is None/missing, default to "CLOSED" (not OPEN)
- Only mark as failed if CB is explicitly OPEN
- Check for all 5 nodes COMPLETED and CB either CLOSED or missing
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from src.pipeline.orchestrator import get_orchestrator
from src.api.models import QueryRequest, QueryResponse
from src.core.circuit_breaker import CircuitBreakerManager, CircuitBreakerConfig
from config.settings import get_settings

logger = logging.getLogger(__name__)

# ============================================================================
# MONITORING DIRECTORY SETUP
# ============================================================================

MONITORING_DIR = Path("./data/monitoring")

def ensure_monitoring_dir():
    """Create monitoring directory if it doesn't exist."""
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

def write_monitoring_json(filename: str, data: Dict[str, Any]) -> Path:
    """Write monitoring data to JSON file."""
    ensure_monitoring_dir()
    filepath = MONITORING_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"📝 Wrote monitoring file: {filepath}")
    return filepath

def read_monitoring_json(filename: str) -> Optional[Dict[str, Any]]:
    """Read monitoring data from JSON file."""
    filepath = MONITORING_DIR / filename
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return None

# ============================================================================
# PROGRESS AGGREGATION FUNCTION (v2.5 NEW)
# ============================================================================

def aggregate_progress_from_nodes(request_id: str) -> tuple[int, str]:
    """
    ✅ NEW v2.5: Calculate progress from node completion status.

    Maps node completion to percentage:
    - 0%: No nodes complete (queued)
    - 20%: Ingestion complete
    - 40%: Preprocessing complete
    - 60%: Chunking complete
    - 80%: Embedding complete
    - 100%: VectorDB complete + CB closed

    Returns: (progress_percent, current_node_name)
    """
    try:
        # 5 nodes in pipeline
        NODES = ["ingestion", "preprocessing", "chunking", "embedding", "vectordb"]
        PROGRESS_PER_NODE = 20  # 100% / 5 nodes = 20% each

        monitoring_path = Path(f"./data/monitoring/nodes/{request_id}")

        if not monitoring_path.exists():
            logger.debug(f"No monitoring path yet for {request_id}")
            return 0, "queued"

        # Count completed nodes
        completed_count = 0
        current_node = "queued"

        for i, node_name in enumerate(NODES):
            node_file = monitoring_path / f"{node_name}_node.json"

            if node_file.exists():
                try:
                    with open(node_file, "r") as f:
                        node_status = json.load(f)

                    # Check if node completed successfully
                    if node_status.get("status") == "COMPLETED":
                        completed_count = i + 1  # This node + all before it
                        if i < len(NODES) - 1:
                            current_node = NODES[i + 1]  # Next node is current
                        else:
                            current_node = NODES[-1]  # Last node
                    elif node_status.get("status") == "PROCESSING":
                        current_node = node_name
                        # Don't increment completed_count; still in progress
                        break
                    elif node_status.get("status") == "FAILED":
                        # Node failed, stop aggregation
                        current_node = node_name
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse {node_file}")
                    continue

        # Calculate progress percentage
        progress = completed_count * PROGRESS_PER_NODE

        # Cap at 100%
        progress = min(progress, 100)

        logger.debug(
            f"📊 Progress aggregation for {request_id}: "
            f"{completed_count} nodes complete → {progress}% "
            f"(current: {current_node})"
        )

        return progress, current_node

    except Exception as e:
        logger.error(f"Error aggregating progress: {e}")
        return 0, "error"

# ============================================================================
# IN-MEMORY INGESTION STORE (API-layer only, NO node coupling)
# ============================================================================

ingestions: Dict[str, Dict[str, Any]] = {}

def create_ingestion(filename, user_name, user_email, metadata):
    """Create new ingestion record."""
    request_id = str(uuid.uuid4())
    ingestions[request_id] = {
        "id": request_id,
        "filename": filename,
        "user_name": user_name,
        "user_email": user_email,
        "status": "queued",
        "progress": 0,
        "current_node": "ingest",
        "node_outputs": {},
        "created_at": datetime.utcnow(),
        # ✅ v2.5: CB state read from pipeline_status.json, not maintained here
        "circuit_breaker_state": "CLOSED",
        "circuit_breaker_reason": None,
    }
    return request_id

def update_progress(request_id, progress, node):
    """Update progress and current node."""
    if request_id in ingestions:
        ingestions[request_id]["progress"] = progress
        ingestions[request_id]["current_node"] = node
        if ingestions[request_id]["status"] == "queued":
            ingestions[request_id]["status"] = "processing"

def check_circuit_breaker_open(request_id) -> bool:
    """Check if circuit breaker is OPEN for this ingestion."""
    if request_id in ingestions:
        cb_state = ingestions[request_id].get("circuit_breaker_state", "CLOSED")
        return cb_state == "OPEN"
    return False

def update_status(request_id, status_value):
    """
    Update ingestion status with CB verification.
    CRITICAL: If circuit breaker is OPEN, ALWAYS set status to "failed"
    """
    if request_id in ingestions:
        cb_is_open = check_circuit_breaker_open(request_id)
        if cb_is_open:
            logger.warning(
                f"⚠️ Circuit breaker OPEN for {request_id} - "
                f"forcing status to 'failed' (was '{status_value}')"
            )
            ingestions[request_id]["status"] = "failed"
        else:
            ingestions[request_id]["status"] = status_value

        current_progress = ingestions[request_id].get("progress", 0)
        current_node = ingestions[request_id].get("current_node", "unknown")

        logger.info(
            f"📊 Status updated: {request_id} → {ingestions[request_id]['status']} "
            f"(progress: {current_progress}%, node: {current_node}, cb_open: {cb_is_open})"
        )
        logger.debug(f"📝 Ingestion state: {ingestions[request_id]}")
    else:
        logger.error(f"❌ Cannot update status: request_id {request_id} not found!")
        logger.error(f"Available IDs: {list(ingestions.keys())}")

def update_circuit_breaker_state(request_id, cb_state: str, reason: Optional[str] = None):
    """
    ✅ v2.5: CB state comes from pipeline_status.json (orchestrator),
    not maintained at API layer anymore.
    """
    if request_id in ingestions:
        ingestions[request_id]["circuit_breaker_state"] = cb_state
        ingestions[request_id]["circuit_breaker_reason"] = reason

        logger.warning(f"🔌 Circuit breaker {cb_state} for {request_id}: {reason}")

        if cb_state == "OPEN":
            logger.error("❌ Circuit breaker OPEN - marking ingestion as FAILED")
            ingestions[request_id]["status"] = "failed"

def add_node_output(request_id, node, output):
    """Add node output to tracking."""
    if request_id in ingestions:
        ingestions[request_id]["node_outputs"][node] = output

def get_ingestion(request_id):
    """Get single ingestion."""
    return ingestions.get(request_id)

def get_all_ingestions_internal():
    """Get all ingestions."""
    return list(ingestions.values())

ingestion_store = type(
    "IngestionStore",
    (),
    {
        "create_ingestion": staticmethod(create_ingestion),
        "update_progress": staticmethod(update_progress),
        "update_status": staticmethod(update_status),
        "add_node_output": staticmethod(add_node_output),
        "get_ingestion": staticmethod(get_ingestion),
        "get_all_ingestions": staticmethod(get_all_ingestions_internal),
        "check_circuit_breaker_open": staticmethod(check_circuit_breaker_open),
        "update_circuit_breaker_state": staticmethod(update_circuit_breaker_state),
    },
)()

circuit_config = CircuitBreakerConfig()
circuit_breaker_manager = CircuitBreakerManager(circuit_config)

# ============================================================================
# ROUTER FACTORY
# ============================================================================

def create_api_router() -> APIRouter:
    """Create and configure the API router."""
    router = APIRouter(prefix="/api", tags=["API"])

    # ====================================================================
    # HEALTH CHECK (GET - unchanged)
    # ====================================================================

    @router.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, str]:
        """Basic health check endpoint."""
        logger.info("📡 Health check called")
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "RAG Pipeline API",
        }

    # ====================================================================
    # INGESTION ENDPOINTS
    # ====================================================================

    @router.post(
        "/ingest/upload",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Ingestion"],
    )
    async def upload_document(
        file: UploadFile = File(...),
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a document for processing.

        Returns:
        - 202 ACCEPTED with ingestion_id
        - success: true (successful upload + pipeline queued)
        """
        try:
            logger.info(f"📤 Upload started: {file.filename} by {user_name}")

            request_id = ingestion_store.create_ingestion(
                file.filename,
                user_name or "unknown",
                user_email or "unknown",
                metadata,
            )

            logger.info(f"✨ Ingestion created: {request_id}")

            # Save file
            upload_dir = Path("./data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / f"{request_id}_{file.filename}"

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            logger.info(f"✅ Document uploaded: {file.filename} → {request_id}")

            # ✅ v2.5 FIXED: Set initial progress to 0 (will aggregate from nodes)
            ingestion_store.update_progress(request_id, 0, "ingest")

            # Queue pipeline processing (NO node coupling here)
            asyncio.create_task(
                process_document_with_orchestrator(
                    request_id=request_id,
                    file_path=str(file_path),
                    filename=file.filename,
                )
            )

            return {
                "success": True,
                "ingestion_id": request_id,
                "file_name": file.filename,
                "status": "queued",
                "message": "Document uploaded and pipeline queued",
            }

        except Exception as e:
            logger.error(f"❌ Upload error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload failed: {str(e)}",
            )

    async def process_document_with_orchestrator(
        request_id: str,
        file_path: str,
        filename: str,
    ) -> None:
        """
        ✅ v2.5 CLEAN: Process document through orchestrator.
        Routes never directly access nodes - only call orchestrator.
        Orchestrator writes monitoring/*.json.
        Routes read monitoring/*.json.
        
        ✅ v2.5-3 FIX: Properly handle CB state when None/missing
        """
        try:
            logger.info(f"🔄 Starting orchestrator pipeline for {request_id}")

            orchestrator = get_orchestrator()

            # Read file
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
                logger.info(f"📄 File loaded: {filename} ({len(file_content)} bytes)")
            except Exception as e:
                logger.error(
                    f"❌ Failed to read file {file_path}: {str(e)}", exc_info=True
                )
                ingestion_store.update_status(request_id, "failed")
                return

            # Process through pipeline (orchestrator handles all the work)
            logger.info(f"🚀 Calling orchestrator.process_document() for {request_id}")

            result_id = await orchestrator.process_document(
                request_id=request_id,
                file_name=filename,
                file_content=file_content,
                metadata={"source_path": file_path},
            )

            logger.debug("📊 Pipeline execution completed, reading monitoring files...")

            # ✅ v2.5: Read pipeline_status.json (single source of truth from orchestrator)
            pipeline_status_file = Path(
                f"./data/monitoring/nodes/{request_id}/pipeline_status.json"
            )

            final_status = None
            final_cb_state = None

            if pipeline_status_file.exists():
                try:
                    with open(pipeline_status_file, "r") as f:
                        ps = json.load(f)

                    final_status = ps.get("status")
                    final_cb_state = ps.get("circuit_breaker_state")

                    logger.info(
                        f"📄 pipeline_status.json → status={final_status}, "
                        f"cb_state={final_cb_state}"
                    )

                except Exception as e:
                    logger.warning(f"⚠️ Could not read pipeline_status.json: {e}")

            # ✅ v2.5 FIXED: Aggregate progress from node files instead of hardcoding
            aggregated_progress, final_node = aggregate_progress_from_nodes(request_id)

            logger.info(
                f"📊 Aggregated progress: {aggregated_progress}% (current node: {final_node})"
            )

            # Sync in-memory store with aggregated progress and pipeline_status.json
            ingestion_store.update_progress(request_id, aggregated_progress, final_node)

            # ✅ v2.5-3 FIX: Handle CB state properly
            # Default CB state to "CLOSED" if missing/None (success case)
            # Only treat as failure if explicitly OPEN or status != completed
            if final_status == "completed":
                # Check if CB is explicitly OPEN (failure)
                # If CB state is None/missing, treat as CLOSED (success)
                if final_cb_state == "OPEN":
                    logger.error(
                        f"❌ Pipeline CB OPEN: "
                        f"pipeline_status.status={final_status}, "
                        f"pipeline_status.cb={final_cb_state}"
                    )
                    ingestion_store.update_status(request_id, "failed")
                    ingestion_store.update_circuit_breaker_state(
                        request_id,
                        "OPEN",
                        "Circuit breaker explicitly OPEN"
                    )
                else:
                    # CB is CLOSED or None/missing - both mean success
                    logger.info(f"✅ Orchestrator completed: {result_id}")
                    ingestion_store.update_progress(request_id, 100, "vectordb")
                    ingestion_store.update_status(request_id, "completed")
                    ingestion_store.update_circuit_breaker_state(
                        request_id,
                        final_cb_state or "CLOSED",
                        None
                    )
                    logger.info(f"🎉 Pipeline COMPLETED: {request_id}")
            else:
                # Status is not "completed" - pipeline failed
                logger.error(
                    "❌ Pipeline not fully successful: "
                    f"pipeline_status.status={final_status}, "
                    f"pipeline_status.cb={final_cb_state}"
                )
                ingestion_store.update_status(request_id, "failed")
                ingestion_store.update_circuit_breaker_state(
                    request_id,
                    final_cb_state or "CLOSED",
                    "Pipeline failed or CB open"
                )

        except Exception as e:
            logger.error(
                f"❌ Pipeline orchestrator error for {request_id}: {str(e)}",
                exc_info=True,
            )

            ingestion_store.update_status(request_id, "failed")

    @router.get("/ingest/status/{ingestion_id}", tags=["Ingestion"])
    async def get_ingestion_status(ingestion_id: str) -> Dict[str, Any]:
        """
        Get real-time ingestion status (aggregated from node files).
        ✅ v2.5: Progress now reflects actual node completion (20% per node)
        """
        try:
            ingestion = ingestion_store.get_ingestion(ingestion_id)

            if not ingestion:
                logger.warning(f"❌ Ingestion not found: {ingestion_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Ingestion {ingestion_id} not found",
                )

            # ✅ v2.5 FIXED: Re-aggregate progress from node files on each status check
            current_progress, current_node = aggregate_progress_from_nodes(ingestion_id)

            # Update in-memory store with latest aggregated progress
            ingestion["progress"] = current_progress
            ingestion["current_node"] = current_node

            logger.info(
                f"✅ Status: {ingestion_id} - "
                f"{current_progress}% - {current_node}"
            )

            return {
                "success": True,
                "status": {
                    "ingestion_id": ingestion_id,
                    "file_name": ingestion["filename"],
                    "progress": current_progress,
                    "current_node": current_node,
                    "status": ingestion["status"],
                    "message": (
                        f"Current stage: {current_node} "
                        f"({current_progress}%)"
                    ),
                    "node_outputs": ingestion.get("node_outputs", {}),
                    "circuit_breaker_state": ingestion.get("circuit_breaker_state", "CLOSED"),
                    "circuit_breaker_reason": ingestion.get("circuit_breaker_reason"),
                    "timestamp": datetime.now().isoformat(),
                },
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Status check failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Status check failed: {str(e)}",
            )

    @router.get("/ingest/all", tags=["Ingestion"])
    async def get_all_ingestions() -> Dict[str, Any]:
        """Get all ingestions with summary metrics."""
        try:
            logger.info("📡 GET /api/ingest/all called")

            all_ing = ingestion_store.get_all_ingestions()
            total = len(all_ing)
            completed = sum(1 for i in all_ing if i["status"] == "completed")
            failed = sum(1 for i in all_ing if i["status"] == "failed")
            in_progress = sum(
                1 for i in all_ing if i["status"] in ["processing", "queued"]
            )

            response = {
                "success": True,
                "summary": {
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                    "in_progress": in_progress,
                },
                "ingestions": [
                    {
                        "ingestion_id": i["id"],
                        "file_name": i["filename"],
                        "status": i["status"],
                        "progress": i["progress"],
                        "current_node": i["current_node"],
                        "circuit_breaker_state": i.get("circuit_breaker_state", "CLOSED"),
                        "created_at": i["created_at"].isoformat() if i["created_at"] else None,
                    }
                    for i in all_ing
                ],
            }

            logger.info(
                "✅ Returned all ingestions: "
                f"total={total}, completed={completed}, "
                f"failed={failed}, in_progress={in_progress}"
            )

            return response

        except Exception as e:
            logger.error(f"❌ Error in get_all_ingestions: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "summary": {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "in_progress": 0,
                },
                "ingestions": [],
            }

    # ====================================================================
    # POST-BASED MONITORING ENDPOINTS (READ from monitoring/*.json files)
    # ====================================================================

    @router.post("/monitor/config", tags=["Monitoring"])
    async def get_pipeline_config() -> Dict[str, Any]:
        """Fetch and persist pipeline configuration."""
        try:
            logger.info("📡 POST /api/monitor/config - Reading pipeline configuration")

            settings = get_settings()

            chunking_strategy = getattr(settings, "chunking_strategy", "recursive")
            embedding_provider = getattr(settings, "embedding_provider", "huggingface")
            embedding_model = getattr(
                settings,
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            vector_db_provider = getattr(settings, "vector_db_provider", "faiss")

            def _val(v: Any) -> str:
                return v.value if hasattr(v, "value") else str(v)

            config_data = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "success",
                "backend_connected": True,
                "configuration": {
                    "chunking": {
                        "strategy": _val(chunking_strategy),
                        "chunk_size": 512,
                        "overlap": 50,
                    },
                    "embeddings": {
                        "provider": _val(embedding_provider),
                        "model": str(embedding_model),
                        "dimension": 384,
                    },
                    "vectordb": {
                        "provider": _val(vector_db_provider),
                        "type": "faiss"
                        if _val(vector_db_provider).lower() == "faiss"
                        else "qdrant",
                        "path": "data/faiss_index"
                        if _val(vector_db_provider).lower() == "faiss"
                        else "qdrant_storage",
                    },
                },
            }

            write_monitoring_json("config.json", config_data)

            logger.info("✅ Pipeline configuration retrieved and persisted")

            return config_data

        except Exception as e:
            logger.error(f"❌ Error in get_pipeline_config: {str(e)}", exc_info=True)

            error_response = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "error",
                "backend_connected": False,
                "error": "Failed to read configuration",
                "details": str(e),
            }

            write_monitoring_json("config.json", error_response)

            return error_response

    @router.post("/monitor/health", tags=["Monitoring"])
    async def get_system_health() -> Dict[str, Any]:
        """Comprehensive system health check."""
        try:
            logger.info("📡 POST /api/monitor/health - Running system health check")

            settings = get_settings()

            vector_db_provider = getattr(settings, "vector_db_provider", "faiss")

            def _val(v: Any) -> str:
                return v.value if hasattr(v, "value") else str(v)

            breaker_status = circuit_breaker_manager.get_all_status()

            all_ing = ingestion_store.get_all_ingestions()

            pipeline_stats = {
                "total_processed": len(all_ing),
                "completed": sum(1 for i in all_ing if i["status"] == "completed"),
                "failed": sum(1 for i in all_ing if i["status"] == "failed"),
                "processing": sum(
                    1 for i in all_ing if i["status"] in ["processing", "queued"]
                ),
            }

            try:
                orchestrator = get_orchestrator()
                orch_health = await orchestrator.health_check()
            except Exception as e:
                logger.warning(f"⚠️ Orchestrator health check failed: {e}")
                orch_health = {"status": "unknown", "error": str(e)}

            health_data = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "success",
                "backend_connected": True,
                "configuration": {
                    "vector_db": _val(vector_db_provider),
                },
                "circuit_breaker": {
                    "overall_state": breaker_status.get("state", "closed"),
                    "total_failures": sum(
                        cb.get("failure_count", 0)
                        for cb in breaker_status.get("breakers", {}).values()
                    ),
                    "breakers": {
                        name: {
                            "state": cb.get("state", "closed"),
                            "failure_count": cb.get("failure_count", 0),
                            "success_count": cb.get("success_count", 0),
                            "last_failure": cb.get("last_failure"),
                        }
                        for name, cb in breaker_status.get("breakers", {}).items()
                    },
                },
                "nodes": {
                    "ingest": {
                        "status": "healthy",
                        "type": "ingestion",
                        "description": "PDF/TXT/JSON parsing",
                        "circuit_breaker": breaker_status.get("breakers", {})
                        .get("ingestion", {})
                        .get("state", "closed"),
                    },
                    "preprocess": {
                        "status": "healthy",
                        "type": "text_cleaning",
                        "description": "Text normalization",
                        "circuit_breaker": breaker_status.get("breakers", {})
                        .get("preprocessing", {})
                        .get("state", "closed"),
                    },
                    "chunk": {
                        "status": "healthy",
                        "type": "text_splitting",
                        "description": "Semantic chunking (512 tokens, 50 overlap)",
                        "circuit_breaker": breaker_status.get("breakers", {})
                        .get("chunking", {})
                        .get("state", "closed"),
                    },
                    "embed": {
                        "status": "healthy",
                        "type": "vectorization",
                        "description": "384-dimensional embeddings",
                        "circuit_breaker": breaker_status.get("breakers", {})
                        .get("embedding", {})
                        .get("state", "closed"),
                    },
                    "upsert": {
                        "status": "healthy",
                        "type": "vector_storage",
                        "description": f"{_val(vector_db_provider)} vector store",
                        "circuit_breaker": breaker_status.get("breakers", {})
                        .get("vectordb", {})
                        .get("state", "closed"),
                    },
                },
                "pipeline": pipeline_stats,
                "orchestrator": orch_health,
            }

            write_monitoring_json("health.json", health_data)

            logger.info("✅ System health check completed and persisted")

            return health_data

        except Exception as e:
            logger.error(f"❌ Error in get_system_health: {str(e)}", exc_info=True)

            error_response = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "error",
                "backend_connected": False,
                "error": "System health check failed",
                "details": str(e),
            }

            write_monitoring_json("health.json", error_response)

            return error_response

    @router.post("/monitor/metrics", tags=["Monitoring"])
    async def get_live_metrics() -> Dict[str, Any]:
        """
        ✅ v2.5: Live pipeline metrics (from in-memory store + aggregated progress).
        Metrics are synced with orchestrator output.
        """
        try:
            logger.info("📡 POST /api/monitor/metrics - Fetching live metrics")

            all_ing = ingestion_store.get_all_ingestions()

            logger.debug(f"📊 Total ingestions in store: {len(all_ing)}")

            completed = sum(1 for i in all_ing if i["status"] == "completed")
            failed = sum(1 for i in all_ing if i["status"] == "failed")
            processing = sum(
                1 for i in all_ing if i["status"] in ["processing", "queued"]
            )

            # ✅ v2.5 FIXED: Re-aggregate progress for each ingestion
            ingestions_with_progress = []

            for i in sorted(all_ing, key=lambda x: x["created_at"], reverse=True):
                current_progress, current_node = aggregate_progress_from_nodes(i["id"])

                ingestions_with_progress.append({
                    "ingestion_id": i["id"],
                    "file_name": i["filename"],
                    "status": i["status"],
                    "progress": current_progress,  # ✅ Aggregated, not static 10%
                    "current_node": current_node,
                    "circuit_breaker_state": i.get("circuit_breaker_state", "CLOSED"),
                    "circuit_breaker_reason": i.get("circuit_breaker_reason"),
                    "created_at": i["created_at"].isoformat() if i["created_at"] else None,
                    "node_outputs": i.get("node_outputs", {}),
                })

            metrics_data = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "success",
                "summary": {
                    "total": len(all_ing),
                    "completed": completed,
                    "failed": failed,
                    "processing": processing,
                },
                "ingestions": ingestions_with_progress,
            }

            logger.info(
                f"✅ Live metrics: total={metrics_data['summary']['total']}, "
                f"completed={metrics_data['summary']['completed']}, "
                f"failed={metrics_data['summary']['failed']}, "
                f"processing={metrics_data['summary']['processing']}"
            )

            write_monitoring_json("metrics.json", metrics_data)

            return metrics_data

        except Exception as e:
            logger.error(f"❌ Error getting metrics: {str(e)}", exc_info=True)

            error_response = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "error",
                "summary": {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "processing": 0,
                },
                "ingestions": [],
                "error": str(e),
            }

            write_monitoring_json("metrics.json", error_response)

            return error_response

    @router.post("/monitor/status", tags=["Monitoring"])
    async def get_full_status() -> Dict[str, Any]:
        """Combined system status (config + health + metrics)."""
        try:
            logger.info("📡 POST /api/monitor/status - Fetching full system status")

            config = read_monitoring_json("config.json") or {}
            health = read_monitoring_json("health.json") or {}
            metrics = read_monitoring_json("metrics.json") or {}

            status_data = {
                "timestamp": datetime.now().isoformat() + "Z",
                "backend_connected": True,
                "status": "operational",
                "config": config.get("configuration", {}),
                "health": {
                    "overall": health.get("status", "unknown"),
                    "circuit_breaker": health.get("circuit_breaker", {}),
                    "nodes": health.get("nodes", {}),
                    "orchestrator": health.get("orchestrator", {}),
                },
                "metrics": {
                    "summary": metrics.get("summary", {}),
                    "ingestion_count": len(metrics.get("ingestions", [])),
                },
                "data_files": {
                    "config": "data/monitoring/config.json",
                    "health": "data/monitoring/health.json",
                    "metrics": "data/monitoring/metrics.json",
                    "status": "data/monitoring/status.json",
                },
            }

            write_monitoring_json("status.json", status_data)

            logger.info("✅ Full system status retrieved and persisted")

            return status_data

        except Exception as e:
            logger.error(f"❌ Error in get_full_status: {str(e)}", exc_info=True)

            error_response = {
                "timestamp": datetime.now().isoformat() + "Z",
                "backend_connected": False,
                "status": "error",
                "error": str(e),
            }

            write_monitoring_json("status.json", error_response)

            return error_response

    @router.post("/monitor/tools-health", tags=["Monitoring"])
    async def get_tools_health() -> Dict[str, Any]:
        """Tools health check for frontend Tools panel."""
        try:
            logger.info("📡 POST /api/monitor/tools-health - Fetching tools health")

            settings = get_settings()

            chunking_strategy = getattr(settings, "chunking_strategy", "recursive")
            embedding_provider = getattr(settings, "embedding_provider", "huggingface")
            embedding_model = getattr(
                settings,
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            vector_db_provider = getattr(settings, "vector_db_provider", "faiss")

            def _val(v):
                return v.value if hasattr(v, "value") else str(v)

            breaker_status = circuit_breaker_manager.get_all_status()

            all_ing = ingestion_store.get_all_ingestions()

            cb_open_count = sum(
                1 for i in all_ing if i.get("circuit_breaker_state") == "OPEN"
            )

            tools_health = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "connected",
                "backend_connected": True,
                "configuration": {
                    "chunking": {
                        "strategy": _val(chunking_strategy),
                        "chunk_size": 512,
                        "overlap": 50,
                    },
                    "embeddings": {
                        "provider": _val(embedding_provider),
                        "model": str(embedding_model),
                        "dimension": 384,
                    },
                    "vectordb": {
                        "provider": _val(vector_db_provider),
                    },
                },
                "circuit_breaker": {
                    "overall_state": breaker_status.get("state", "closed"),
                    "ingestions_with_open_cb": cb_open_count,
                    "breakers": {
                        name: {
                            "state": cb.get("state", "closed"),
                            "failure_count": cb.get("failure_count", 0),
                            "success_count": cb.get("success_count", 0),
                        }
                        for name, cb in breaker_status.get("breakers", {}).items()
                    },
                },
                "nodes": {
                    "ingest": {
                        "status": "healthy",
                        "description": "PDF/TXT/JSON parsing",
                        "circuit_breaker_state": breaker_status.get("breakers", {})
                        .get("ingestion", {})
                        .get("state", "closed"),
                    },
                    "preprocess": {
                        "status": "healthy",
                        "description": "Text normalization",
                        "circuit_breaker_state": breaker_status.get("breakers", {})
                        .get("preprocessing", {})
                        .get("state", "closed"),
                    },
                    "chunk": {
                        "status": "healthy",
                        "description": "Semantic chunking (512 tokens, 50 overlap)",
                        "circuit_breaker_state": breaker_status.get("breakers", {})
                        .get("chunking", {})
                        .get("state", "closed"),
                    },
                    "embed": {
                        "status": "healthy",
                        "description": "384-dimensional embeddings",
                        "circuit_breaker_state": breaker_status.get("breakers", {})
                        .get("embedding", {})
                        .get("state", "closed"),
                    },
                    "upsert": {
                        "status": "healthy",
                        "description": f"{_val(vector_db_provider)} vector store",
                        "circuit_breaker_state": breaker_status.get("breakers", {})
                        .get("vectordb", {})
                        .get("state", "closed"),
                    },
                },
            }

            write_monitoring_json("tools_health.json", tools_health)

            logger.info("✅ Tools health retrieved")

            return tools_health

        except Exception as e:
            logger.error(f"❌ Error in tools_health: {str(e)}", exc_info=True)

            error = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "error",
                "backend_connected": False,
                "error": "Failed to fetch tools health",
                "details": str(e),
                "configuration": {},
                "circuit_breaker": {},
                "nodes": {},
            }

            write_monitoring_json("tools_health.json", error)

            return error

    @router.post("/monitor/logs", tags=["Monitoring"])
    async def get_pipeline_logs(
        level: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Fetch live pipeline logs."""
        try:
            from src.core.log_buffer import get_log_buffer

            logger.info("📡 POST /api/monitor/logs - Fetching pipeline logs")

            log_buffer = get_log_buffer()

            logs = log_buffer.get_logs(
                level=level,
                search=search,
                limit=limit,
            )

            summary = log_buffer.get_summary()

            logs_response = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "success",
                "summary": summary,
                "filters": {
                    "level": level or "all",
                    "search": search or "none",
                    "limit": limit,
                },
                "results": {
                    "total_fetched": len(logs),
                    "logs": logs,
                },
                "data_file": "data/monitoring/pipeline_logs.json",
            }

            write_monitoring_json("logs.json", logs_response)

            logger.info(f"✅ Pipeline logs retrieved: {len(logs)} logs")

            return logs_response

        except Exception as e:
            logger.error(f"❌ Error in get_pipeline_logs: {str(e)}", exc_info=True)

            error_response = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "error",
                "error": "Failed to fetch logs",
                "details": str(e),
                "results": {
                    "total_fetched": 0,
                    "logs": [],
                },
            }

            write_monitoring_json("logs.json", error_response)

            return error_response

    @router.post("/query", response_model=QueryResponse, tags=["Query"])
    async def query(request: QueryRequest) -> QueryResponse:
        """Query the RAG system."""
        try:
            if not request.query or len(request.query) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Query too short (minimum 2 characters)",
                )

            orchestrator = get_orchestrator()

            results = await orchestrator.query_documents(
                query=request.query,
                top_k=request.topk or 5,
                session_id=request.session_id,
            )

            logger.info(f"✅ Query executed: {request.query} - {len(results)} results")

            return QueryResponse(results=results)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {str(e)}",
            )

    return router
