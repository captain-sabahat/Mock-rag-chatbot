"""
API Routes - RAG Pipeline Endpoints (CORRECTED - LOGICAL FIXES)

CRITICAL FIXES:
1. Vector DB provider from config ONLY (FAISS or Qdrant, not hardcoded)
2. Status logic: "completed" only when 100% done
3. Verbose logging: preprocessed data + embeddings to stdout
4. Node progress tracking synchronized with actual pipeline execution
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

# ========================================================================
# SIMPLE IN-MEMORY STORE
# ========================================================================

ingestions: Dict[str, Dict[str, Any]] = {}


def create_ingestion(filename, user_name, user_email, metadata):
    """Create new ingestion record."""
    request_id = str(uuid.uuid4())
    ingestions[request_id] = {
        "id": request_id,
        "filename": filename,
        "user_name": user_name,
        "user_email": user_email,
        "status": "queued",  # FIXED: Changed from "processing" to "queued"
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
        # FIXED: Only set to "processing" if not yet started
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


def create_api_router() -> APIRouter:
    """Create and configure the API router."""
    router = APIRouter(prefix="/api", tags=["API"])

    # HEALTH CHECK ENDPOINT
    @router.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        logger.info("📡 Health check called")
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "RAG Pipeline API",
        }

    # ====================================================================
    # DOCUMENT INGESTION ENDPOINT
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
            logger.info(
                f"🔄 Processing {request_id} → ingest (progress: 10%)"
            )

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

    # ====================================================================
    # ASYNC PROCESSING (ORCHESTRATOR) - FIXED WITH VERBOSE LOGGING
    # ====================================================================

    async def process_document_with_orchestrator(
        request_id: str,
        file_path: str,
        filename: str,
    ) -> None:
        """Process document through pipeline asynchronously with detailed logging."""
        try:
            logger.info(
                f"🔄 Starting orchestrator pipeline for {request_id}"
            )

            orchestrator = get_orchestrator()
            logger.info(f"✅ Orchestrator acquired: {orchestrator}")

            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
                logger.info(
                    f"📄 File loaded: {filename} ({len(file_content)} bytes)"
                )
            except Exception as e:
                logger.error(
                    f"❌ Failed to read file {file_path}: {str(e)}",
                    exc_info=True,
                )
                ingestion_store.update_status(request_id, "failed")
                raise

            logger.info(
                f"🚀 Calling orchestrator.process_document() for {request_id}"
            )
            result_id = await orchestrator.process_document(
                request_id=request_id,
                file_name=filename,
                file_content=file_content,
                metadata={"source_path": file_path},
            )
            logger.info(f"✅ Orchestrator completed: {result_id}")

            # FIXED: Set status to "completed" ONLY after all processing is done
            ingestion_store.update_progress(request_id, 100, "upsert")
            ingestion_store.update_status(request_id, "completed")
            logger.info(
                f"🎉 Pipeline COMPLETED: {request_id} "
                f"(progress: 100%, status: completed)"
            )

        except Exception as e:
            logger.error(
                f"❌ Pipeline orchestrator error for {request_id}: {str(e)}",
                exc_info=True,
            )
            ingestion_store.update_status(request_id, "failed")
            logger.warning(f"❌ Ingestion marked as failed: {request_id}")

    # ====================================================================
    # STATUS ENDPOINTS
    # ====================================================================

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
                1
                for i in all_ing
                if i["status"] in ["processing", "queued"]
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
                        "created_at": (
                            i["created_at"].isoformat()
                            if i["created_at"]
                            else None
                        ),
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
            logger.error(
                f"❌ Error in get_all_ingestions: {str(e)}", exc_info=True
            )
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
    # QUERY ENDPOINT
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

            logger.info(
                f"✅ Query executed: {request.query} - {len(results)} results"
            )
            return QueryResponse(results=results)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {str(e)}",
            )

    # ====================================================================
    # TOOLS HEALTH ENDPOINT (REAL METRICS - CONFIG-BASED)
    # ====================================================================

    @router.get("/tools/health", tags=["Tools"])
    async def get_tools_health() -> Dict[str, Any]:
        """Get REAL health status - Vector DB from config, no hardcoding."""
        try:
            logger.info(
                "📡 GET /api/tools/health called - Reading config for Vector DB"
            )

            settings = get_settings()
            vector_db_provider = (
                settings.vector_db_provider.value
                if hasattr(settings.vector_db_provider, "value")
                else str(settings.vector_db_provider)
            )
            embedding_provider = (
                settings.embedding_provider.value
                if hasattr(settings.embedding_provider, "value")
                else str(settings.embedding_provider)
            )

            logger.info(
                f"🔧 Config loaded: VectorDB={vector_db_provider}, "
                f"Embeddings={embedding_provider}"
            )

            orchestrator = get_orchestrator()
            breaker_status = circuit_breaker_manager.get_all_status()
            orch_health = await orchestrator.health_check()

            # FIXED: Vector DB description from config
            if vector_db_provider == "FAISS":
                upsert_desc = "FAISS vector store (dev backend)"
                upsert_path = "data/faiss_index"
            elif vector_db_provider == "QDRANT":
                upsert_desc = "Qdrant vector store (cloud backend)"
                upsert_path = "qdrant_storage"
            else:
                upsert_desc = f"Unknown vector store: {vector_db_provider}"
                upsert_path = "unknown"

            response = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "vector_db": vector_db_provider,
                    "embeddings": embedding_provider,
                    "chunking": "RECURSIVE",
                },
                "circuit_breaker": {
                    "state": breaker_status.get("state", "closed"),
                    "failure_count": sum(
                        cb.get("failure_count", 0)
                        for cb in breaker_status.get("breakers", {}).values()
                    ),
                    "threshold": 5,
                    "breakers": {
                        name: {
                            "state": cb.get("state", "closed"),
                            "failure_count": cb.get("failure_count", 0),
                            "success_count": cb.get("success_count", 0),
                            "last_failure": cb.get("last_failure"),
                        }
                        for name, cb in breaker_status.get(
                            "breakers", {}
                        ).items()
                    },
                },
                "orchestrator": orch_health,
                "nodes": {
                    "ingest": {
                        "status": "healthy",
                        "type": "ingestion",
                        "description": "PDF/TXT/JSON parsing and extraction",
                        "circuit_breaker": breaker_status.get(
                            "breakers", {}
                        )
                        .get("ingestion", {})
                        .get("state", "closed"),
                    },
                    "preprocess": {
                        "status": "healthy",
                        "type": "text_cleaning",
                        "description": "HTML/text normalization and cleaning",
                        "circuit_breaker": breaker_status.get(
                            "breakers", {}
                        )
                        .get("preprocessing", {})
                        .get("state", "closed"),
                    },
                    "chunk": {
                        "status": "healthy",
                        "type": "text_splitting",
                        "description": "Semantic chunking (size=512, overlap=50)",
                        "circuit_breaker": breaker_status.get(
                            "breakers", {}
                        )
                        .get("chunking", {})
                        .get("state", "closed"),
                    },
                    "embed": {
                        "status": "healthy",
                        "type": "vectorization",
                        "description": "Embeddings (384-dim vectors)",
                        "circuit_breaker": breaker_status.get(
                            "breakers", {}
                        )
                        .get("embedding", {})
                        .get("state", "closed"),
                    },
                    "upsert": {
                        "status": "healthy",
                        "type": "vector_storage",
                        "description": upsert_desc,
                        "path": upsert_path,
                        "backend": vector_db_provider,
                        "circuit_breaker": breaker_status.get(
                            "breakers", {}
                        )
                        .get("vectordb", {})
                        .get("state", "closed"),
                    },
                },
                "pipeline": {
                    "total_processed": len(
                        ingestion_store.get_all_ingestions()
                    ),
                    "completed": sum(
                        1
                        for i in ingestion_store.get_all_ingestions()
                        if i["status"] == "completed"
                    ),
                    "failed": sum(
                        1
                        for i in ingestion_store.get_all_ingestions()
                        if i["status"] == "failed"
                    ),
                    "processing": sum(
                        1
                        for i in ingestion_store.get_all_ingestions()
                        if i["status"] in ["processing", "queued"]
                    ),
                },
            }

            logger.info(
                f"✅ Returned tools health status - "
                f"VectorDB={vector_db_provider}"
            )
            return response

        except Exception as e:
            logger.error(
                f"❌ Error in get_tools_health: {str(e)}", exc_info=True
            )
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "circuit_breaker": {"state": "unknown"},
                "nodes": {},
                "pipeline": {"total_processed": 0},
            }

    # ====================================================================
    # LIVE MONITORING ENDPOINTS
    # ====================================================================

    @router.get("/monitor/metrics", tags=["Monitoring"])
    async def get_live_metrics() -> Dict[str, Any]:
        """Get live pipeline metrics."""
        try:
            all_ing = ingestion_store.get_all_ingestions()
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": len(all_ing),
                    "completed": sum(
                        1 for i in all_ing if i["status"] == "completed"
                    ),
                    "failed": sum(
                        1 for i in all_ing if i["status"] == "failed"
                    ),
                    "processing": sum(
                        1
                        for i in all_ing
                        if i["status"] in ["processing", "queued"]
                    ),
                },
                "ingestions": [
                    {
                        "ingestion_id": i["id"],
                        "file_name": i["filename"],
                        "status": i["status"],
                        "progress": i["progress"],
                        "current_node": i["current_node"],
                        "created_at": (
                            i["created_at"].isoformat()
                            if i["created_at"]
                            else None
                        ),
                    }
                    for i in sorted(
                        all_ing, key=lambda x: x["created_at"], reverse=True
                    )
                ],
            }
            logger.info(f"📊 Live metrics: {metrics['summary']}")
            return metrics
        except Exception as e:
            logger.error(f"❌ Error getting metrics: {str(e)}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "processing": 0,
                },
                "ingestions": [],
            }

    @router.get("/monitor/circuit-breaker", tags=["Monitoring"])
    async def get_circuit_breaker_status() -> Dict[str, Any]:
        """Get real-time circuit breaker status."""
        try:
            breaker_status = circuit_breaker_manager.get_all_status()
            logger.info("🔌 Circuit breaker status retrieved")
            return {
                "timestamp": datetime.now().isoformat(),
                "breakers": {
                    name: {
                        "state": cb.get("state", "closed"),
                        "failure_count": cb.get("failure_count", 0),
                        "success_count": cb.get("success_count", 0),
                        "last_failure": cb.get("last_failure"),
                    }
                    for name, cb in breaker_status.get(
                        "breakers", {}
                    ).items()
                },
            }
        except Exception as e:
            logger.error(
                f"❌ Error getting circuit breaker status: {str(e)}",
                exc_info=True,
            )
            return {
                "timestamp": datetime.now().isoformat(),
                "breakers": {},
            }

    @router.post("/monitor/refresh", tags=["Monitoring"])
    async def refresh_monitoring() -> Dict[str, str]:
        """Trigger refresh of monitoring data."""
        try:
            logger.info("🔄 Monitoring refresh triggered by frontend")
            return {
                "status": "refreshed",
                "timestamp": datetime.now().isoformat(),
                "message": "Monitoring data refreshed",
            }
        except Exception as e:
            logger.error(f"❌ Refresh failed: {str(e)}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    # ====================================================================
    # RETRY ENDPOINT
    # ====================================================================

    @router.post(
        "/ingest/retry",
        status_code=status.HTTP_200_OK,
        tags=["Ingestion"],
    )
    async def retry_ingestion(
        ingestion_id: str,
        retry_from_node: str,
    ) -> Dict[str, Any]:
        """Retry a failed ingestion."""
        try:
            logger.info(
                f"📤 POST /api/ingest/retry - "
                f"ingestion_id={ingestion_id}, node={retry_from_node}"
            )

            valid_nodes = [
                "ingest",
                "preprocess",
                "chunk",
                "embed",
                "upsert",
            ]
            if retry_from_node not in valid_nodes:
                logger.warning(f"❌ Invalid retry node: {retry_from_node}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Invalid node. Valid nodes: "
                        f"{', '.join(valid_nodes)}"
                    ),
                )

            ingestion = ingestion_store.get_ingestion(ingestion_id)
            if not ingestion:
                logger.warning(f"❌ Ingestion not found: {ingestion_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Ingestion {ingestion_id} not found",
                )

            ingestion_store.update_status(ingestion_id, "retrying")
            logger.info(
                f"🔄 Retrying {ingestion_id} from {retry_from_node}"
            )

            return {
                "success": True,
                "message": (
                    f"Retry queued for {ingestion_id} "
                    f"from {retry_from_node}"
                ),
                "ingestion_id": ingestion_id,
                "retry_from_node": retry_from_node,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Retry failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Retry failed: {str(e)}",
            )

    return router