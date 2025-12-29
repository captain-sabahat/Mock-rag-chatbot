# src/api/routes.py

import logging
import asyncio
import time
import os
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Request,
    UploadFile,
    File,
    Form,
)

from src.pipeline.schemas import (
    UserQueryRequest,
    RAGResponse,
    ResponseStatus,
    PipelineLogic,
)

from src.core.exceptions import RAGPipelineException, RecoverableException
from src.api.dependencies import (
    get_settings,
    get_orchestrator,
    get_request_context,
    get_logger_context,
)

from src.utils import (
    generate_request_id,
    extract_text_from_file,
    hash_document,
)
from src.config.settings import Settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])

# ============================================================================
# PHASE 1: EXISTING ENDPOINT (UNTOUCHED)
# ============================================================================

@router.post(
    "/query",
    response_model=RAGResponse,
    summary="User query endpoint",
    description="Submit user query with optional document attachment",
)
async def query_endpoint(
    request: UserQueryRequest,
    orchestrator=Depends(get_orchestrator),
    request_context: dict = Depends(get_request_context),
    logger_context: dict = Depends(get_logger_context),
) -> RAGResponse:
    """
    Main user query endpoint.
    Processes user query through RAG pipeline.
    Returns answer with sources and metadata.
    
    WORKFLOW:
    1. Extract user_id and session_id from request
    2. Initialize context (request_id, logging)
    3. Call orchestrator.execute(request)
    4. Orchestrator:
       - Gets tools from ServiceContainer (LLM, SLM, Embeddings, VectorDB)
       - Calls logic_router (determine LOGIC1/2/3/4)
       - Executes appropriate pipeline
       - Returns RAGResponse
    5. Return response to client
    """
    start_time = time.time()
    request_id = request_context.get("request_id", generate_request_id())

    try:
        logger.info(
            f"Query received [{request_id}]",
            extra={
                "request_id": request_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "doc_attached": (
                    request.doc_attached.value if request.doc_attached else None
                ),
            },
        )

        # Validate prompt length
        if len(request.prompt) > 5000:
            logger.warning(
                f"Prompt too long [{request_id}]: {len(request.prompt)} chars"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt too long (max 5000 characters)",
            )

        # Validate document size if attached
        if (
            request.doc_attached
            and hasattr(request, "document")
            and request.document
        ):
            if hasattr(request.document, "file_size"):
                if request.document.file_size > 52428800:  # 50MB
                    logger.warning(
                        f"Document too large [{request_id}]: "
                        f"{request.document.file_size} bytes"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Document too large (max 50MB)",
                    )

        logger.debug(f"Calling orchestrator [{request_id}]")

        # Call orchestrator with timeout protection
        try:
            rag_response = await asyncio.wait_for(
                orchestrator.execute(request),
                timeout=30.0,  # Max 30s for any query
            )
        except asyncio.TimeoutError:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"Query timeout [{request_id}] after {processing_time_ms}ms"
            )
            return RAGResponse(
                status=ResponseStatus.ERROR,
                error="Query processing timeout (30s)",
                logic_path=PipelineLogic.LOGIC2,
                request_id=request_id,
            )

        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Query completed [{request_id}]",
            extra={
                "request_id": request_id,
                "user_id": request.user_id,
                "logic_path": rag_response.logic_path,
                "processing_time_ms": processing_time_ms,
            },
        )

        # Add correlation ID to response
        rag_response.request_id = request_id
        return rag_response

    except RAGPipelineException as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"RAG error [{request_id}] - {e.error_code}",
            extra={
                "request_id": request_id,
                "error_code": e.error_code,
                "error_message": e.message,
                "processing_time_ms": processing_time_ms,
            },
        )

        return RAGResponse(
            status=ResponseStatus.ERROR,
            error=e.message,
            logic_path=PipelineLogic.LOGIC2,
            request_id=request_id,
        )

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, etc.)
        raise

    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Unexpected error [{request_id}]: {str(e)}",
            extra={
                "request_id": request_id,
                "processing_time_ms": processing_time_ms,
            },
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


# ============================================================================
# PHASE 2: EPHEMERAL DOCUMENT UPLOAD ENDPOINT (CORRECTED)
# ============================================================================
# CRITICAL: Document → SLM Summarization → Augmentation/Generation ONLY
# NO chunking, NO embedding, NO vector DB storage

# Constants for file validation
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
MAX_FILE_SIZE_BYTES = 52428800  # 50 MB
CHUNK_READ_SIZE = 8192  # 8KB chunks for streaming validation


async def _validate_file_extension(filename: str) -> None:
    """
    Validate file has allowed extension.
    
    Args:
        filename: File name to validate
    
    Raises:
        HTTPException(400): If extension not allowed
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )


async def _validate_file_size(file: UploadFile) -> None:
    """
    Validate file size by streaming chunks.
    Prevents loading huge files into memory.
    Rewinds file pointer after validation.
    
    Args:
        file: UploadFile object to validate
    
    Raises:
        HTTPException(413): If file exceeds size limit
    """
    total_bytes = 0

    # Read in chunks to avoid loading full file into memory
    chunk = await file.read(CHUNK_READ_SIZE)
    while chunk:
        total_bytes += len(chunk)
        if total_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    "File too large. Max allowed: "
                    f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB"
                ),
            )
        chunk = await file.read(CHUNK_READ_SIZE)

    # Rewind file pointer for downstream processing
    await file.seek(0)


@router.post(
    "/upload",
    status_code=200,
    summary="Upload document for ephemeral summarization & Q&A",
    description=(
        "Upload a document with a query for instant summarization via SLM. "
        "Document is NOT stored persistently or in vector DB. "
        "Summary is used for answering query via generation pipeline."
    ),
)
async def upload_and_answer(
    file: UploadFile = File(...),
    query: str = Form(..., min_length=5, max_length=5000),
    user_id: str = Form("user_default"),
    session_id: str = Form("session_default"),
    orchestrator=Depends(get_orchestrator),
    settings: Settings = Depends(get_settings),
    request_context: dict = Depends(get_request_context),
) -> dict:
    """
    CORRECT: Ephemeral document upload + query endpoint.
    
    FLOW:
    1. File validation (extension, size)
    2. Extract text from file (PDF/DOCX/TXT/MD)
    3. Call SLM summarizer → get document summary
    4. Use summary + session history → generate answer
    5. NO chunking, NO vector DB storage, NO embedding persistence
    
    FEATURES:
    - POST only (no GET)
    - Requires both document AND query
    - Validates file type (pdf, txt, md, docx)
    - Validates file size (<= 50 MB)
    - Validates query length (5-5000 chars)
    - Summary used for this query only
    - Session history (conversation) is persistent per session_id
    - Document itself is ephemeral/temporary
    
    Args:
        file: UploadFile (required)
        query: User question (5-5000 chars, required)
        user_id: User identifier (optional, default: "user_default")
        session_id: Session/chat ID (optional, default: "session_default")
        orchestrator: Dependency injection
        settings: Dependency injection
        request_context: Dependency injection
    
    Returns:
        {
            "status": "success",
            "answer": "Generated answer text",
            "sources": ["original document filename"],
            "metadata": {
                "filename": "document.pdf",
                "user_id": "user123",
                "session_id": "session456",
                "ephemeral": True,
                "latency_ms": 2543,
                "request_id": "..."
            }
        }
    """
    request_id = request_context.get("request_id", generate_request_id())
    start_time = time.time()

    try:
        logger.info(
            f"Ephemeral doc upload received [{request_id}]: {file.filename}",
            extra={
                "request_id": request_id,
                "filename": file.filename,
                "user_id": user_id,
                "session_id": session_id,
                "operation": "ephemeral_doc_upload",
            },
        )

        # -------- SECURITY: File Validation --------

        # 1. Validate filename exists
        if not file.filename:
            logger.warning(f"Upload rejected [{request_id}]: no filename")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required",
            )

        # 2. Validate file extension
        await _validate_file_extension(file.filename)
        logger.debug(f"File extension valid [{request_id}]: {file.filename}")

        # 3. Validate file size (streaming, memory-safe)
        await _validate_file_size(file)
        logger.debug(f"File size valid [{request_id}]")

        # 4. Validate query
        if not query or len(query.strip()) < 5:
            logger.warning(
                f"Upload rejected [{request_id}]: query too short ({len(query)} chars)"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be at least 5 characters",
            )

        query = query.strip()
        logger.debug(f"Query valid [{request_id}]: {len(query)} chars")

        # -------- READ FILE SAFELY --------
        # Read file bytes into memory
        # Safe because we validated size above
        content_bytes = await file.read()
        await file.close()

        logger.info(
            f"File read successfully [{request_id}]: {len(content_bytes)} bytes",
            extra={
                "request_id": request_id,
                "file_size_bytes": len(content_bytes),
            },
        )

        # -------- STEP 1: Extract Text from File --------
        logger.debug(f"Extracting text from file [{request_id}]: {file.filename}")

        _, file_extension = os.path.splitext(file.filename.lower())

        try:
            extracted_text = extract_text_from_file(
                content=content_bytes,
                filename=file.filename,
                extension=file_extension,
            )
        except Exception as e:
            logger.error(
                f"Text extraction failed [{request_id}]: {str(e)}",
                extra={"request_id": request_id},
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to extract text: {str(e)}",
            )

        if not extracted_text or len(extracted_text.strip()) == 0:
            logger.warning(f"No text extracted [{request_id}]: {file.filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document contains no readable text",
            )

        logger.info(
            f"Text extracted [{request_id}]: {len(extracted_text)} chars",
            extra={
                "request_id": request_id,
                "extracted_chars": len(extracted_text),
            },
        )

        # -------- CALL ORCHESTRATOR --------
        # Pass extracted text to orchestrator's answer_with_ephemeral_doc
        logger.debug(
            f"Calling orchestrator.answer_with_ephemeral_doc() [{request_id}]"
        )

        try:
            result = await asyncio.wait_for(
                orchestrator.answer_with_ephemeral_doc(
                    query=query,
                    document_text=extracted_text,
                    filename=file.filename,
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
                ),
                timeout=30.0,  # Max 30s for doc Q&A
            )

        except asyncio.TimeoutError:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"Document processing timeout [{request_id}] after {processing_time_ms}ms"
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Document processing timeout (30s)",
            )

        # -------- RETURN SUCCESS --------
        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Ephemeral doc processed successfully [{request_id}]",
            extra={
                "request_id": request_id,
                "filename": file.filename,
                "user_id": user_id,
                "session_id": session_id,
                "processing_time_ms": processing_time_ms,
                "operation": "ephemeral_doc_upload",
            },
        )

        return {
            "status": "success",
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "metadata": {
                "filename": file.filename,
                "user_id": user_id,
                "session_id": session_id,
                "ephemeral": True,
                "latency_ms": processing_time_ms,
                "request_id": request_id,
            },
        }

    except HTTPException:
        # Already logged and formatted
        raise

    except RAGPipelineException as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"RAG pipeline error [{request_id}]: {e.error_code}",
            extra={
                "request_id": request_id,
                "error_code": e.error_code,
                "error_message": e.message,
                "processing_time_ms": processing_time_ms,
                "operation": "ephemeral_doc_upload",
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {e.message}",
        )

    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Unexpected error during upload [{request_id}]: {str(e)}",
            extra={
                "request_id": request_id,
                "filename": file.filename if file else "unknown",
                "processing_time_ms": processing_time_ms,
                "operation": "ephemeral_doc_upload",
            },
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process uploaded document",
        )
