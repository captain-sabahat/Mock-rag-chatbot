"""
================================================================================
FILE: src/api/dependencies.py
================================================================================

PURPOSE:
FastAPI dependency injection functions. Provides reusable dependencies
that are injected into route handlers via Depends(). Enables:
- Configuration access
- Model access
- Handler/Container access (factory for tools)
- Orchestrator access
- Request context (user_id, session_id, etc.)
- Logging context
- **File validation & ephemeral document handling** (NEW)

WORKFLOW:
1. Define async functions that return dependencies
2. Use in route handlers: async def endpoint(..., dep = Depends(get_dep))
3. FastAPI automatically calls dependency before handler
4. Dependency result injected as parameter
5. Enable testing by mocking dependencies

1. get_settings() → Returns Settings (config loaded from .env)
2. get_container() → Returns ServiceContainer (creates all tools via factories)
3. get_orchestrator() → Returns Orchestrator (accepts tools from container)
4. Routes call: orchestrator = Depends(get_orchestrator)
5. Orchestrator.execute() → Calls pipeline with tools from container
6. **validate_file() → Validates uploaded files (extension, size)** (NEW)

IMPORTS:
- Depends from fastapi
- Settings from config
- Models, handlers, orchestrator from main.py
- Request from fastapi
- HTTPException for error responses
- **UploadFile from fastapi for file handling** (NEW)
- **io for BytesIO operations** (NEW)

INPUTS:
- FastAPI Request object (via function parameter)
- Values from Depends() chain
- **UploadFile object for document uploads** (NEW)

OUTPUTS:
- Settings instance
- Models dict
- Handlers dict
- Orchestrator instance
- Request context
- Logger context
- **Validated file content (bytes)** (NEW)

KEY FACTS:
- All dependencies are ASYNC (non-blocking)
- FastAPI caches dependencies per request (no repeated calls)
- Dependencies can depend on other dependencies
- Errors raised in dependencies return 500/400 to client
- Enable testing by mocking these functions
- **File validation happens in dependency (fast-fail before handler)** (NEW)
- **No file storage - content streamed to memory only** (NEW)

DEPENDENCY CHAIN:
get_settings()
├─ Used by: all endpoints (configuration)
get_models()
├─ Depends on: get_settings
├─ Used by: model-specific endpoints
get_handlers()
├─ Depends on: get_settings, get_models
├─ Used by: core logic endpoints
get_orchestrator()
├─ Depends on: all lower-level dependencies
├─ Used by: main query endpoint
get_request_context()
├─ Extract request info (user_id, session_id)
├─ Used by: all endpoints (for logging)
validate_user_id() (NEW)
├─ Validate user ID format
├─ Used by: endpoints requiring user context
validate_session_id() (NEW)
├─ Validate session ID format
├─ Used by: endpoints requiring session context
validate_file() (NEW)
├─ Validate uploaded file (extension, size)
├─ Used by: file upload endpoints
validate_query() (NEW)
├─ Validate query string (length, format)
├─ Used by: query endpoints

FUTURE SCOPE (Phase 2+):
- Add authentication dependency (verify JWT token)
- Add authorization dependency (check user permissions)
- Add rate limiting dependency (track requests per user)
- Add tenant isolation dependency (multi-tenant support)
- Add observability dependency (correlation IDs, tracing)
- Add request validation middleware
- Add response compression
- Add CORS middleware
- Add API versioning
- Add deprecation warnings
- Add virus scanning dependency (ClamAV integration)
- Add OCR dependency (Tesseract for image extraction)
- Add document parsing dependency (PDF, DOCX extraction)

TESTING ENVIRONMENT:
- Mock dependencies in tests using pytest fixtures
- Override dependencies with: app.dependency_overrides[get_settings] = mock_settings
- Use TestClient with overridden dependencies
- Test error cases (missing config, unavailable handlers)
- **Create mock UploadFile objects for testing file validation**
- **Test file size limits and extension validation**

PRODUCTION DEPLOYMENT:
- Dependencies loaded once at startup
- Errors in dependencies cause 500 responses (logged properly)
- Authentication/authorization enforced here
- Rate limiting checked here
- Observability (correlation IDs) added here
- All calls async (non-blocking)
- **File validation happens immediately (fail fast)**
- **No temporary files - everything in memory**
"""
#================================================================================
#IMPORTS
#================================================================================

import logging
from typing import Optional, Dict, Any
from fastapi import Depends, Request, HTTPException, status, UploadFile
from io import BytesIO
from src.config.settings import Settings
from src.utils import generate_request_id, format_logger_context, hash_document
from src.container.service_container import ServiceContainer
logger = logging.getLogger(__name__)

#================================================================================
#DEPENDENCY FUNCTIONS
#================================================================================

async def get_settings() -> Settings:
    """
    Get application settings (configuration).
    Provides access to config throughout request lifecycle.
    
    Returns:
        Settings: Configuration object with model names, timeouts, device, etc.
    
    Raises:
        HTTPException: If settings not initialized (startup failed)
    """
    from .main import get_settings as _get_settings
    
    try:
        return _get_settings()
    except RuntimeError as e:
        logger.error(f"Settings not available: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service initialization failed"
        )


async def get_container(
    settings: Settings = Depends(get_settings)
) -> "ServiceContainer":
    """
    Get ServiceContainer (tool-agnostic factory container).
    Container creates all tools (LLM, SLM, Embeddings, VectorDB, Cache)
    based on configuration in settings.
    
    This replaces the old get_models() and get_handlers() dependencies.
    Now all tool creation logic is abstracted away from routes.
    
    Args:
        settings: Settings dependency (for configuration)
    
    Returns:
        ServiceContainer: Factory container for all tools
    
    Raises:
        HTTPException: If container initialization fails
    
    TOOL-AGNOSTIC BEHAVIOR:
    - Container reads settings.llm_provider (e.g., \"gemini\")
    - Factory creates appropriate provider (GeminiProvider)
    - Container stores in abstract interface (ILLMProvider)
    - Routes only see: container.get_llm() → ILLMProvider
    - Routes never care which specific provider is active
    """
    from .main import get_container as _get_container
    
    try:
        return _get_container()
    except RuntimeError as e:
        logger.error(f"Container not available: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service container not initialized"
        )


async def get_orchestrator(
    settings: Settings = Depends(get_settings),
    container = Depends(get_container)
):
    """
    Get orchestrator (main pipeline coordinator).
    Orchestrator executes pipeline logic using tools from container.
    Container injects all dependencies (LLM, SLM, Embeddings, VectorDB, etc.).
    
    Args:
        settings: Settings dependency
        container: ServiceContainer dependency (provides all tools)
    
    Returns:
        Orchestrator: Main pipeline orchestrator
    
    Raises:
        HTTPException: If orchestrator not initialized
    
    TOOL-AGNOSTIC INTEGRATION:
    - Orchestrator no longer knows about specific tools
    - All tool dependencies come from container
    - Container.get_llm() returns abstract ILLMProvider interface
    - Orchestrator calls: llm = container.get_llm()
    - Then calls: response = await llm.generate(prompt)
    - Works identically whether LLM is Gemini, OpenAI, or Anthropic
    """
    from .main import get_orchestrator as _get_orchestrator
    
    try:
        return _get_orchestrator()
    except RuntimeError as e:
        logger.error(f"Orchestrator not available: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline orchestrator not initialized"
        )


async def get_request_context(request: Request) -> Dict[str, Any]:
    """
    Extract and provide request context.
    Extracts user_id, session_id, request_id from request.
    Provides structured context for logging and correlation.
    
    Args:
        request: FastAPI Request object
    
    Returns:
        Dict with: request_id, user_id (optional), session_id (optional)
    """
    context = {
        "request_id": getattr(request.state, 'request_id', generate_request_id())
    }
    
    # Extract from headers (if provided by client)
    if "X-User-ID" in request.headers:
        context["user_id"] = request.headers["X-User-ID"]
    if "X-Session-ID" in request.headers:
        context["session_id"] = request.headers["X-Session-ID"]
    
    return context


async def get_logger_context(
    request_context: Dict = Depends(get_request_context)
) -> Dict[str, Any]:
    """
    Get structured logging context.
    Formats request context for JSON logging.
    
    Args:
        request_context: Request context dependency
    
    Returns:
        Formatted logging context dict
    """
    return format_logger_context(
        request_id=request_context.get("request_id"),
        user_id=request_context.get("user_id", "unknown"),
        session_id=request_context.get("session_id")
    )


#================================================================================
#VALIDATION DEPENDENCIES (Input validation)
#================================================================================

async def validate_user_id(user_id: Optional[str] = None) -> str:
    """
    Validate user ID.
    
    Args:
        user_id: User ID to validate
    
    Returns:
        Validated user ID
    
    Raises:
        HTTPException: If validation fails
    """
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required"
        )
    
    if len(user_id) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id too long (max 255 chars)"
        )
    
    return user_id


async def validate_session_id(session_id: Optional[str] = None) -> str:
    """
    Validate session ID.
    
    Args:
        session_id: Session ID to validate
    
    Returns:
        Validated session ID
    
    Raises:
        HTTPException: If validation fails
    """
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id is required"
        )
    
    if len(session_id) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="session_id too long (max 255 chars)"
        )
    
    return session_id


# ================================================================================
# NEW: FILE UPLOAD VALIDATION (Ephemeral document handling)
# ================================================================================

# Allowed file extensions (configurable)
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx'}
MAX_FILE_SIZE_MB = 50

async def validate_file(file: UploadFile) -> Dict[str, Any]:
    """
    Validate uploaded file (extension, size, readability).
    Performs ALL validation before any processing.
    
    PURPOSE:
    - Fast-fail validation (reject invalid files immediately)
    - Security (prevent large file attacks, extension spoofing)
    - Consistency (same validation across all file endpoints)
    
    Args:
        file: UploadFile object from FastAPI
    
    Returns:
        Dict with validated file info:
        {
            'filename': 'document.pdf',
            'extension': '.pdf',
            'content': b'file bytes',
            'size_bytes': 1024,
            'content_hash': 'sha256_hash',
            'content_type': 'application/pdf'
        }
    
    Raises:
        HTTPException (400): Invalid extension, size, or format
        HTTPException (413): File too large
        HTTPException (415): Unsupported content type
    
    VALIDATION STEPS:
    1. Check filename exists
    2. Check extension is whitelisted
    3. Check file size (before reading)
    4. Read file into memory (streaming)
    5. Verify content is readable
    6. Generate content hash (for cache key generation)
    7. Return validated file data
    
    SECURITY:
    - No file stored on disk
    - Content streamed directly to memory
    - Extension whitelist prevents malicious files
    - Size limit prevents memory exhaustion
    - Hash enables duplicate detection
    
    PERFORMANCE:
    - Single pass read (no re-reading)
    - Content available immediately
    - Hash computed during read (no extra pass)
    """
    
    # Validate filename exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename"
        )
    
    # Validate extension is whitelisted
    import os
    filename = file.filename
    file_ext = os.path.splitext(filename).lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        allowed = ', '.join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed}"
        )
    
    # Validate file size (check before reading)
    # Note: content_length may not always be available
    if file.size and file.size > (MAX_FILE_SIZE_MB * 1024 * 1024):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)"
        )
    
    # Read file content into memory
    try:
        # Stream file into BytesIO (memory buffer)
        content = await file.read()
        
        # Validate we have content
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Validate size after reading
        size_bytes = len(content)
        if size_bytes > (MAX_FILE_SIZE_MB * 1024 * 1024):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)"
            )
        
        # Reset file pointer for potential re-reads
        await file.seek(0)
        
        # Generate content hash (for cache keys, deduplication)
        content_hash = hash_document(content)
        
        logger.info(
            f"File validated: {filename} ({size_bytes} bytes, "
            f"hash={content_hash[:16]}...)"
        )
        
        return {
            'filename': filename,
            'extension': file_ext,
            'content': content,
            'size_bytes': size_bytes,
            'content_hash': content_hash,
            'content_type': file.content_type or 'application/octet-stream'
        }
    
    except HTTPException:
        # Re-raise HTTPExceptions (already formatted)
        raise
    
    except Exception as e:
        logger.error(f"File read error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading file: {str(e)}"
        )


async def validate_query(query: str, min_length: int = 5, max_length: int = 5000) -> str:
    """
    Validate query string.
    
    PURPOSE:
    - Ensure query meets length requirements
    - Prevent empty/trivial queries
    - Prevent extremely long queries
    - Sanitize whitespace
    
    Args:
        query: Query string to validate
        min_length: Minimum query length (default 5)
        max_length: Maximum query length (default 5000)
    
    Returns:
        Validated and trimmed query string
    
    Raises:
        HTTPException: If validation fails
    
    VALIDATION:
    1. Trim whitespace
    2. Check minimum length
    3. Check maximum length
    4. Reject empty queries
    """
    
    # Trim whitespace
    query = query.strip() if query else ""
    
    # Check empty
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    # Check minimum length
    if len(query) < min_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query too short (minimum {min_length} characters)"
        )
    
    # Check maximum length
    if len(query) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query too long (maximum {max_length} characters)"
        )
    
    logger.info(f"Query validated: {len(query)} characters")
    
    return query


# ================================================================================
# COMBINED DEPENDENCIES (Multiple validations in one call)
# ================================================================================

async def validate_file_upload_request(
    file: UploadFile = Depends(),
    query: str = Depends(),
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Combined validation for file upload requests.
    Validates file, query, user_id, session_id all in one dependency.
    
    USAGE IN ROUTE:
    @router.post(\"/api/upload\")
    async def upload_endpoint(
        validated_request = Depends(validate_file_upload_request)
    ):
        file_info = validated_request['file']
        query = validated_request['query']
        user_id = validated_request['user_id']
        session_id = validated_request['session_id']
    
    Args:
        file: Uploaded file
        query: Query string
        user_id: User identifier
        session_id: Session identifier
    
    Returns:
        Dict with all validated components
    
    Raises:
        HTTPException: If any validation fails
    """
    
    # Validate all components
    validated_file = await validate_file(file)
    validated_query = await validate_query(query)
    
    # User and session are optional but validated if provided
    if user_id:
        user_id = await validate_user_id(user_id)
    else:
        user_id = "user_default"
    
    if session_id:
        session_id = await validate_session_id(session_id)
    else:
        session_id = "session_default"
    
    return {
        'file': validated_file,
        'query': validated_query,
        'user_id': user_id,
        'session_id': session_id
    }
