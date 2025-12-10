# ============================================================================
# API Routes - RAG Pipeline Endpoints (POST-Based Monitoring)
# ============================================================================

"""
API Routes - RAG Pipeline Endpoints with POST-based monitoring

Key Changes:
1. All monitoring endpoints changed from GET to POST
2. POST endpoints write JSON to data/monitoring/ directory
3. Safe attribute access using getattr() - no AttributeError
4. Structured response format for all endpoints
5. Circuit breaker tracking integrated
6. Ingestion pipeline metrics tracked
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
# IN-MEMORY INGESTION STORE
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
    }
    return request_id

def update_progress(request_id, progress, node):
    """Update progress and current node."""
    if request_id in ingestions:
        ingestions[request_id]["progress"] = progress
        ingestions[request_id]["current_node"] = node
        if ingestions[request_id]["status"] == "queued":
            ingestions[request_id]["status"] = "processing"

def update_status(request_id, status_value):
    """Update ingestion status."""
    if request_id in ingestions:
        ingestions[request_id]["status"] = status_value
        logger.info(
            f"📊 Status updated: {request_id} → {status_value} "
            f"(progress: {ingestions[request_id]['progress']}%)"
        )

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
    # INGESTION ENDPOINTS (unchanged)
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
        """Upload a document for processing."""
        try:
            logger.info(f"📤 Upload started: {file.filename} by {user_name}")
            request_id = ingestion_store.create_ingestion(
                file.filename,
                user_name or "unknown",
                user_email or "unknown",
                metadata,
            )

            logger.info(f"✨ Ingestion created: {request_id}")
            upload_dir = Path("./data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / f"{request_id}_{file.filename}"

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            logger.info(f"✅ Document uploaded: {file.filename} → {request_id}")
            ingestion_store.update_progress(request_id, 10, "ingest")

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
        """Process document through pipeline asynchronously."""
        try:
            logger.info(f"🔄 Starting orchestrator pipeline for {request_id}")
            orchestrator = get_orchestrator()

            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
                logger.info(f"📄 File loaded: {filename} ({len(file_content)} bytes)")
            except Exception as e:
                logger.error(f"❌ Failed to read file {file_path}: {str(e)}", exc_info=True)
                ingestion_store.update_status(request_id, "failed")
                raise

            logger.info(f"🚀 Calling orchestrator.process_document() for {request_id}")
            result_id = await orchestrator.process_document(
                request_id=request_id,
                file_name=filename,
                file_content=file_content,
                metadata={"source_path": file_path},
            )

            logger.info(f"✅ Orchestrator completed: {result_id}")
            ingestion_store.update_progress(request_id, 100, "upsert")
            ingestion_store.update_status(request_id, "completed")
            logger.info(f"🎉 Pipeline COMPLETED: {request_id}")

        except Exception as e:
            logger.error(f"❌ Pipeline orchestrator error for {request_id}: {str(e)}", exc_info=True)
            ingestion_store.update_status(request_id, "failed")

    @router.get("/ingest/status/{ingestion_id}", tags=["Ingestion"])
    async def get_ingestion_status(ingestion_id: str) -> Dict[str, Any]:
        """Get real-time ingestion status."""
        try:
            ingestion = ingestion_store.get_ingestion(ingestion_id)
            if not ingestion:
                logger.warning(f"❌ Ingestion not found: {ingestion_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Ingestion {ingestion_id} not found",
                )

            logger.info(
                f"✅ Status: {ingestion_id} - "
                f"{ingestion['progress']}% - {ingestion['current_node']}"
            )

            return {
                "success": True,
                "status": {
                    "ingestion_id": ingestion_id,
                    "file_name": ingestion["filename"],
                    "progress": ingestion["progress"],
                    "current_node": ingestion["current_node"],
                    "status": ingestion["status"],
                    "message": (
                        f"Current stage: {ingestion['current_node']} "
                        f"({ingestion['progress']}%)"
                    ),
                    "node_outputs": ingestion.get("node_outputs", {}),
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
    # POST-BASED MONITORING ENDPOINTS (NEW)
    # ====================================================================

    @router.post("/monitor/config", tags=["Monitoring"])
    async def get_pipeline_config() -> Dict[str, Any]:
        """
        POST endpoint to fetch and persist current pipeline configuration.
        Reads from .env via settings.
        Writes to data/monitoring/config.json
        """
        try:
            logger.info("📡 POST /api/monitor/config - Reading pipeline configuration")
            
            settings = get_settings()

            # SAFE access using getattr - never crashes
            chunking_strategy = getattr(settings, "chunking_strategy", "recursive")
            embedding_provider = getattr(settings, "embedding_provider", "huggingface")
            embedding_model = getattr(
                settings,
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            vector_db_provider = getattr(settings, "vector_db_provider", "faiss")

            # Helper to extract enum values
            def _val(v: Any) -> str:
                return v.value if hasattr(v, "value") else str(v)

            config_data = {
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
                        "type": "faiss" if _val(vector_db_provider).lower() == "faiss" else "qdrant",
                        "path": "data/faiss_index" if _val(vector_db_provider).lower() == "faiss" else "qdrant_storage",
                    },
                },
            }

            # Write to persistent JSON
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
        """
        POST endpoint for comprehensive system health check.
        Returns:
        - Circuit breaker status for each node
        - Pipeline metrics (total, completed, failed, processing)
        - Orchestrator health
        - Configuration status
        
        Writes to data/monitoring/health.json
        """
        try:
            logger.info("📡 POST /api/monitor/health - Running system health check")

            settings = get_settings()
            
            # Safe access
            vector_db_provider = getattr(settings, "vector_db_provider", "faiss")
            def _val(v: Any) -> str:
                return v.value if hasattr(v, "value") else str(v)

            # Circuit breaker status
            breaker_status = circuit_breaker_manager.get_all_status()

            # Pipeline metrics
            all_ing = ingestion_store.get_all_ingestions()
            pipeline_stats = {
                "total_processed": len(all_ing),
                "completed": sum(1 for i in all_ing if i["status"] == "completed"),
                "failed": sum(1 for i in all_ing if i["status"] == "failed"),
                "processing": sum(1 for i in all_ing if i["status"] in ["processing", "queued"]),
            }

            # Orchestrator health (if available)
            try:
                orchestrator = get_orchestrator()
                orch_health = await orchestrator.health_check()
            except Exception as e:
                logger.warning(f"⚠️ Orchestrator health check failed: {e}")
                orch_health = {"status": "unknown", "error": str(e)}

            health_data = {
                "timestamp": datetime.now().isoformat() + "Z",
                "status": "healthy",
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
                        "circuit_breaker": breaker_status.get("breakers", {}).get("ingestion", {}).get("state", "closed"),
                    },
                    "preprocess": {
                        "status": "healthy",
                        "type": "text_cleaning",
                        "description": "Text normalization",
                        "circuit_breaker": breaker_status.get("breakers", {}).get("preprocessing", {}).get("state", "closed"),
                    },
                    "chunk": {
                        "status": "healthy",
                        "type": "text_splitting",
                        "description": "Semantic chunking (512 tokens, 50 overlap)",
                        "circuit_breaker": breaker_status.get("breakers", {}).get("chunking", {}).get("state", "closed"),
                    },
                    "embed": {
                        "status": "healthy",
                        "type": "vectorization",
                        "description": "384-dimensional embeddings",
                        "circuit_breaker": breaker_status.get("breakers", {}).get("embedding", {}).get("state", "closed"),
                    },
                    "upsert": {
                        "status": "healthy",
                        "type": "vector_storage",
                        "description": f"{_val(vector_db_provider)} vector store",
                        "circuit_breaker": breaker_status.get("breakers", {}).get("vectordb", {}).get("state", "closed"),
                    },
                },
                "pipeline": pipeline_stats,
                "orchestrator": orch_health,
            }

            # Write to persistent JSON
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
        POST endpoint for live pipeline metrics and ingestion tracking.
        Returns detailed breakdown of all ingestions.
        Writes to data/monitoring/metrics.json
        """
        try:
            logger.info("📡 POST /api/monitor/metrics - Fetching live metrics")

            all_ing = ingestion_store.get_all_ingestions()

            metrics_data = {
                "timestamp": datetime.now().isoformat() + "Z",
                "summary": {
                    "total": len(all_ing),
                    "completed": sum(1 for i in all_ing if i["status"] == "completed"),
                    "failed": sum(1 for i in all_ing if i["status"] == "failed"),
                    "processing": sum(1 for i in all_ing if i["status"] in ["processing", "queued"]),
                },
                "ingestions": [
                    {
                        "ingestion_id": i["id"],
                        "file_name": i["filename"],
                        "status": i["status"],
                        "progress": i["progress"],
                        "current_node": i["current_node"],
                        "created_at": i["created_at"].isoformat() if i["created_at"] else None,
                        "node_outputs": i.get("node_outputs", {}),
                    }
                    for i in sorted(all_ing, key=lambda x: x["created_at"], reverse=True)
                ],
            }

            # Write to persistent JSON
            write_monitoring_json("metrics.json", metrics_data)

            logger.info(f"✅ Live metrics: {metrics_data['summary']}")
            return metrics_data

        except Exception as e:
            logger.error(f"❌ Error getting metrics: {str(e)}", exc_info=True)
            error_response = {
                "timestamp": datetime.now().isoformat() + "Z",
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
        """
        POST endpoint for combined system status.
        Aggregates config + health + metrics into single response.
        Writes to data/monitoring/status.json
        """
        try:
            logger.info("📡 POST /api/monitor/status - Fetching full system status")

            # Read existing files
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

            # Write to persistent JSON
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

    # ====================================================================
    # QUERY ENDPOINT (unchanged)
    # ====================================================================

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
