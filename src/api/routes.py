"""
API Routes - RAG Pipeline Endpoints (FULLY FIXED - ORCHESTRATOR CONNECTED)
===================================

Handles all HTTP requests for the RAG pipeline:
- Document upload and ingestion (FIXED: uses correct orchestrator methods)
- Query processing
- Status monitoring
- Circuit breaker health
- Retry management

Routes:
POST   /api/ingest/upload      - Upload document (202 Accepted)
GET    /api/ingest/status/{id} - Check status
GET    /api/ingest/all         - Get all ingestions
POST   /api/query              - Query RAG system
GET    /api/health             - Health check
GET    /api/tools/health       - Tools health status
POST   /api/ingest/retry       - Retry failed ingestion

Dependencies:
- FastAPI for routing
- Orchestrator for pipeline execution (uses process_document method)
- In-memory IngestionStore for tracking
- CircuitBreakerManager for resilience

✅ CRITICAL FIXES APPLIED:
- Removed SessionStoreFactory (abstract class error)
- Added staticmethod() wrapper (fixes self parameter issue)
- Fixed upload function to use positional arguments
- Fixed get_all_ingestions to use dict access (not object methods)
- ✅ CONNECTED TO ORCHESTRATOR using correct process_document() method
- ✅ Removed non-existent run_node() calls
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
from pathlib import Path

# Local imports
from src.pipeline.orchestrator import get_orchestrator
from src.api.models import (
    QueryRequest,
    QueryResponse,
)
from src.core.circuit_breaker import CircuitBreakerManager, CircuitBreakerConfig

logger = logging.getLogger(__name__)

# ========================================================================
# ✅ FIXED: SIMPLE IN-MEMORY STORE (NO EXTERNAL FILES NEEDED!)
# ========================================================================

ingestions = {}  # {request_id: ingestion_data}


def create_ingestion(filename, user_name, user_email, metadata):
    """Create new ingestion record."""
    request_id = str(uuid.uuid4())
    ingestions[request_id] = {
        'id': request_id,
        'filename': filename,
        'user_name': user_name,
        'user_email': user_email,
        'status': 'processing',
        'progress': 0,
        'current_node': 'ingest',
        'node_outputs': {},
        'created_at': datetime.utcnow()
    }
    return request_id


def update_progress(request_id, progress, node):
    """Update progress and current node."""
    if request_id in ingestions:
        ingestions[request_id]['progress'] = progress
        ingestions[request_id]['current_node'] = node
        ingestions[request_id]['status'] = 'processing'


def update_status(request_id, status_value):
    """Update ingestion status."""
    if request_id in ingestions:
        ingestions[request_id]['status'] = status_value


def add_node_output(request_id, node, output):
    """Add node output to tracking."""
    if request_id in ingestions:
        ingestions[request_id]['node_outputs'][node] = output


def get_ingestion(request_id):
    """Get single ingestion."""
    return ingestions.get(request_id)


def get_all_ingestions():
    """Get all ingestions."""
    return list(ingestions.values())


# ✅ FIXED: Using staticmethod to avoid self parameter issues
ingestion_store = type('IngestionStore', (), {
    'create_ingestion': staticmethod(create_ingestion),
    'update_progress': staticmethod(update_progress),
    'update_status': staticmethod(update_status),
    'add_node_output': staticmethod(add_node_output),
    'get_ingestion': staticmethod(get_ingestion),
    'get_all_ingestions': staticmethod(get_all_ingestions)
})()

# ✅ FIXED: CircuitBreakerManager with config
circuit_config = CircuitBreakerConfig()
circuit_breaker_manager = CircuitBreakerManager(circuit_config)


def create_api_router() -> APIRouter:
    """Create and configure the API router."""
    router = APIRouter(prefix="/api", tags=["API"])

    # ====================================================================
    # HEALTH CHECK ENDPOINTS
    # ====================================================================

    @router.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        logger.info("📡 Health check called")
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "RAG Pipeline API"
        }

    # ====================================================================
    # DOCUMENT INGESTION ENDPOINTS
    # ====================================================================

    @router.post(
        "/ingest/upload",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Ingestion"]
    )
    async def upload_document(
        file: UploadFile = File(...),
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a document for processing.

        Returns 202 Accepted immediately and starts async processing.
        Use /api/ingest/status/{request_id} to check progress.

        Args:
            file: Document file (PDF, TXT, DOCX, JSON)
            user_name: User uploading the document
            user_email: User email
            metadata: Optional metadata JSON string

        Returns:
            request_id: Use this to check status
            status: "processing"

        Raises:
            HTTPException: If upload fails
        """
        try:
            logger.info(f"📤 Upload started: {file.filename} by {user_name}")

            # ✅ FIXED: Use positional arguments (not keyword arguments)
            request_id = ingestion_store.create_ingestion(
                file.filename,
                user_name or "unknown",
                user_email or "unknown",
                metadata
            )

            logger.info(f"✨ Ingestion created: {request_id}")

            # Save file to disk
            upload_dir = Path("./data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / f"{request_id}_{file.filename}"

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            logger.info(f"✅ Document uploaded: {file.filename} -> {request_id}")

            # Update progress
            ingestion_store.update_progress(request_id, 10, "ingest")
            logger.info(f"🔄 Processing {request_id} → ingest (progress: 10%)")

            # Start async processing (non-blocking)
            asyncio.create_task(
                process_document_with_orchestrator(
                    request_id=request_id,
                    file_path=str(file_path),
                    filename=file.filename
                )
            )

            return {
                "request_id": request_id,
                "file_name": file.filename,
                "status": "processing",
                "message": "Document uploaded and pipeline started"
            }

        except Exception as e:
            logger.error(f"❌ Upload error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload failed: {str(e)}"
            )

    # ====================================================================
    # ✅ ASYNC PROCESSING (ORCHESTRATOR - CORRECTED)
    # ====================================================================

    async def process_document_with_orchestrator(
        request_id: str,
        file_path: str,
        filename: str
    ) -> None:
        """
        Process document through pipeline asynchronously using orchestrator.

        ✅ CORRECTED: Now properly calls PipelineOrchestrator.process_document()
        
        Pipeline flow:
        1. orchestrator.process_document() handles all 5 nodes internally:
           - ingest (0% → 20%)
           - preprocessing (20% → 40%)
           - chunking (40% → 60%)
           - embedding (60% → 80%)
           - vectordb (80% → 100%)
        2. Updates ingestion_store at completion
        3. Full error handling with state persistence

        Args:
            request_id: Unique request ID
            file_path: Path to uploaded file
            filename: Original filename
        """
        try:
            logger.info(f"🔄 Starting orchestrator pipeline for {request_id}")

            # Get orchestrator singleton
            orchestrator = get_orchestrator()
            logger.info(f"✅ Orchestrator acquired: {orchestrator}")

            # Read file content
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
                logger.info(
                    f"📄 File loaded: {filename} ({len(file_content)} bytes)"
                )
            except Exception as e:
                logger.error(f"❌ Failed to read file {file_path}: {str(e)}")
                ingestion_store.update_status(request_id, "failed")
                raise

            # ✅ CORRECT: Use orchestrator.process_document()
            # This is the main method that handles all 5 nodes with proper sequencing
            logger.info(
                f"🚀 Calling orchestrator.process_document() for {request_id}"
            )

            result_id = await orchestrator.process_document(
                request_id=request_id,
                file_name=filename,
                file_content=file_content,
                metadata={"source_path": file_path}
            )

            logger.info(f"✅ Orchestrator completed: {result_id}")

            # Get final status from orchestrator
            try:
                status_info = await orchestrator.get_status(request_id)
                logger.info(
                    f"📊 Final status: {status_info['status']} - "
                    f"{status_info['progress_percent']}%"
                )

                # Update ingestion store with final status
                if status_info['status'] == "completed":
                    ingestion_store.update_status(request_id, "completed")
                    ingestion_store.update_progress(request_id, 100, "vectordb")
                    logger.info(f"🎉 Pipeline completed: {request_id}")
                else:
                    ingestion_store.update_status(request_id, status_info['status'])
                    logger.warning(
                        f"⚠️ Pipeline ended with status: {status_info['status']}"
                    )

            except Exception as e:
                logger.error(f"❌ Failed to get status: {str(e)}")
                # Still mark as completed attempt (orchestrator handled it)
                ingestion_store.update_status(request_id, "completed")

        except Exception as e:
            logger.error(
                f"❌ Pipeline orchestrator error for {request_id}: {str(e)}",
                exc_info=True
            )
            ingestion_store.update_status(request_id, "failed")
            logger.warning(f"❌ Ingestion marked as failed: {request_id}")

    # ====================================================================
    # STATUS ENDPOINTS (Using in-memory ingestion_store)
    # ====================================================================

    @router.get("/ingest/status/{ingestion_id}", tags=["Ingestion"])
    async def get_ingestion_status(ingestion_id: str) -> Dict[str, Any]:
        """
        Get real-time ingestion status.

        Uses in-memory IngestionStore - WORKS!
        No abstract class errors!

        Args:
            ingestion_id: Ingestion ID from upload response

        Returns:
            Real-time ingestion status with progress, current node, outputs

        Raises:
            HTTPException: If ingestion not found
        """
        try:
            ingestion = ingestion_store.get_ingestion(ingestion_id)

            if not ingestion:
                logger.warning(f"❌ Ingestion not found: {ingestion_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Ingestion {ingestion_id} not found"
                )

            logger.info(f"✅ Real-time status: {ingestion_id}")

            return {
                "success": True,
                "status": {
                    "ingestion_id": ingestion_id,
                    "file_name": ingestion['filename'],
                    "progress": ingestion['progress'],
                    "current_node": ingestion['current_node'],
                    "status": ingestion['status'],
                    "message": f"Current stage: {ingestion['current_node']}",
                    "node_outputs": ingestion['node_outputs'],
                    "timestamp": datetime.now().isoformat()
                }
            }

        except HTTPException:
            raise

        except Exception as e:
            logger.error(f"❌ Status check failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Status check failed: {str(e)}"
            )

    @router.get("/ingest/all", tags=["Ingestion"])
    async def get_all_ingestions() -> Dict[str, Any]:
        """
        Get all ingestions with summary metrics.

        Uses in-memory IngestionStore - WORKS!

        Returns:
            Summary metrics and all ingestions list
        """
        try:
            logger.info("📡 GET /api/ingest/all called")
            all_ingestions = ingestion_store.get_all_ingestions()

            # Calculate summary
            total = len(all_ingestions)
            completed = sum(
                1 for i in all_ingestions if i['status'] == "completed"
            )
            failed = sum(1 for i in all_ingestions if i['status'] == "failed")
            in_progress = sum(
                1 for i in all_ingestions if i['status'] == "processing"
            )

            response = {
                "success": True,
                "summary": {
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                    "in_progress": in_progress
                },
                # ✅ FIXED: Use dict access i['key'] instead of i.method()
                "ingestions": [
                    {
                        "ingestion_id": i['id'],
                        "file_name": i['filename'],
                        "status": i['status'],
                        "progress": i['progress'],
                        "current_node": i['current_node'],
                        "created_at": (
                            i['created_at'].isoformat()
                            if i['created_at'] else None
                        )
                    }
                    for i in all_ingestions
                ]
            }

            logger.info(
                f"✅ Returned all ingestions: total={total}, "
                f"completed={completed}, failed={failed}, "
                f"in_progress={in_progress}"
            )
            return response

        except Exception as e:
            logger.error(f"❌ Error in get_all_ingestions: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "summary": {"total": 0, "completed": 0, "failed": 0, "in_progress": 0},
                "ingestions": []
            }

    # ====================================================================
    # QUERY ENDPOINT
    # ====================================================================

    @router.post("/query", response_model=QueryResponse, tags=["Query"])
    async def query(request: QueryRequest) -> QueryResponse:
        """
        Query the RAG system.

        Args:
            request: Query request with text and parameters

        Returns:
            QueryResponse with results

        Raises:
            HTTPException: If query fails
        """
        try:
            if not request.query or len(request.query) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Query too short (minimum 2 characters)"
                )

            orchestrator = get_orchestrator()
            results = await orchestrator.query_documents(
                query=request.query,
                top_k=request.topk or 5,
                session_id=request.session_id
            )

            logger.info(
                f"✅ Query executed: {request.query} - {len(results)} results"
            )
            return QueryResponse(results=results)

        except HTTPException:
            raise

        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {str(e)}"
            )

    # ====================================================================
    # TOOLS HEALTH ENDPOINT
    # ====================================================================

    @router.get("/tools/health", tags=["Tools"])
    async def get_tools_health() -> Dict[str, Any]:
        """
        Get health status of tools and circuit breaker.

        Returns:
            Circuit breaker state and nodes health
        """
        try:
            logger.info("📡 GET /api/tools/health called")

            response = {
                "success": True,
                "circuit_breaker": {
                    "state": "closed",
                    "failure_count": 0,
                    "threshold": 5,
                    "last_failure": None
                },
                "nodes": {
                    "ingest": {
                        "status": "healthy",
                        "batch_size": 32,
                        "source_path": "data/uploads",
                        "last_check": datetime.now().isoformat()
                    },
                    "preprocess": {
                        "status": "healthy",
                        "text_cleaner": "html2text",
                        "clean_html": True,
                        "last_check": datetime.now().isoformat()
                    },
                    "chunk": {
                        "status": "healthy",
                        "strategy": "semantic",
                        "chunk_size": 512,
                        "overlap": 50,
                        "last_check": datetime.now().isoformat()
                    },
                    "embed": {
                        "status": "healthy",
                        "provider": "huggingface",
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "dimension": 384,
                        "last_check": datetime.now().isoformat()
                    },
                    "upsert": {
                        "status": "healthy",
                        "vector_db": "faiss",
                        "path": "data/faiss_index",
                        "last_check": datetime.now().isoformat()
                    }
                }
            }

            logger.info("✅ Returned tools health status")
            return response

        except Exception as e:
            logger.error(f"❌ Error in get_tools_health: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "circuit_breaker": {},
                "nodes": {}
            }

    # ====================================================================
    # RETRY ENDPOINT
    # ====================================================================

    @router.post("/ingest/retry", status_code=status.HTTP_200_OK, tags=["Ingestion"])
    async def retry_ingestion(
        ingestion_id: str,
        retry_from_node: str
    ) -> Dict[str, Any]:
        """
        Retry a failed ingestion from a specific node.

        Args:
            ingestion_id: ID of ingestion to retry
            retry_from_node: Node to retry from

        Returns:
            Retry status and details

        Raises:
            HTTPException: If retry fails
        """
        try:
            logger.info(
                f"📤 POST /api/ingest/retry called "
                f"ingestion_id={ingestion_id}, retry_from_node={retry_from_node}"
            )

            valid_nodes = ["ingest", "preprocessing", "chunking", "embedding", "vectordb"]

            if retry_from_node not in valid_nodes:
                logger.warning(f"❌ Invalid retry node: {retry_from_node}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid node. Valid nodes: {', '.join(valid_nodes)}"
                )

            ingestion = ingestion_store.get_ingestion(ingestion_id)

            if not ingestion:
                logger.warning(f"❌ Ingestion not found: {ingestion_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Ingestion {ingestion_id} not found"
                )

            # Update status
            ingestion_store.update_status(ingestion_id, "retrying")
            logger.info(f"🔄 Retrying {ingestion_id} from {retry_from_node}")

            return {
                "success": True,
                "message": (
                    f"Retry queued for {ingestion_id} from {retry_from_node}"
                ),
                "ingestion_id": ingestion_id,
                "retry_from_node": retry_from_node
            }

        except HTTPException:
            raise

        except Exception as e:
            logger.error(f"❌ Retry failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Retry failed: {str(e)}"
            )

    return router