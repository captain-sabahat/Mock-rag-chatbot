# MERGED: 5 sections with separation comments
#│   │   ├── SECTION 1: Enumerations (RedisLookupFlag, DocAttachedFlag, PipelineLogic)
#│   │   ├── SECTION 2: UserQuery schemas
#│   │   ├── SECTION 3: RAGResult schemas
#│   │   ├── SECTION 4: Session schemas
#│   │   └── SECTION 5: Error schemas
# merged files api/request_models.py and api/response_models.py
"""
================================================================================
FILE: src/schemas.py
================================================================================

PURPOSE:
    Pydantic data models for request/response validation, session management,
    and error handling. Single source of truth for all data structures in the
    RAG backend. Enables automatic validation, type hints, and API documentation.

WORKFLOW:
    1. SECTION 1: Enumerations (flags, statuses)
    2. SECTION 2: UserQuery request schemas (from api/request_models.py)
    3. SECTION 3: RAGResult response schemas (from api/response_models.py)
    4. SECTION 4: Session management schemas
    5. SECTION 5: Error response schemas

IMPORTS:
    - pydantic: Data validation
    - datetime: Timestamp handling
    - typing: Type hints
    - enum: Enumeration support

INPUTS (via API requests):
    - UserQueryRequest (prompt, document, flags)
    - DocumentAttachment (file metadata)

OUTPUTS (API responses):
    - RAGResponse (status, result, error)
    - RAGResult (answer, sources, summary)
    - ErrorResponse (error details)

KEY FACTS:
    - All models are Pydantic BaseModel (automatic validation)
    - Used by FastAPI for request/response serialization
    - Enables automatic API documentation (OpenAPI/Swagger)
    - Type hints for IDE autocomplete
    - Validation rules enforced at model boundaries

VALIDATION RULES:
    - UserQueryRequest.prompt: non-empty string
    - UserQueryRequest.user_id: non-empty string
    - UserQueryRequest.session_id: non-empty string
    - DocumentAttachment.file_size: max 50MB
    - RAGResponse.status: one of ["success", "error"]

FUTURE SCOPE (Phase 2+):
    - Add custom validators (email, phone, etc.)
    - Add computed fields (derived from other fields)
    - Add model serialization customization
    - Add OpenAPI documentation customization
    - Add request/response examples
    - Add versioning support (multiple schema versions)
    - Add deprecation warnings
    - Add field descriptions (for API docs)
    - Add validation metrics tracking
    - Add schema migration support

TESTING ENVIRONMENT:
    - Create test instances with test data
    - Validate schema constraints
    - Test invalid inputs (should raise ValidationError)
    - Test serialization/deserialization

PRODUCTION DEPLOYMENT:
    - Schemas loaded at startup (zero runtime overhead)
    - Validation happens at API boundary
    - Clear error messages for invalid input
"""

# ================================================================================
# IMPORTS
# ================================================================================

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# ================================================================================
# SECTION 1: ENUMERATIONS (Flags and statuses)
# ================================================================================

class RedisLookupFlag(str, Enum):
    """
    Flag indicating whether to check Redis cache for query results.
    
    YES: Check cache first (LOGIC 1 fast path)
    NO: Skip cache, go directly to RAG pipeline
    """
    YES = "yes"
    NO = "no"


class DocAttachedFlag(str, Enum):
    """
    Flag indicating whether user attached a document.
    
    YES: Document attached (use LINE 1 - summarization)
    NO: No document (skip summarization)
    """
    YES = "yes"
    NO = "no"


class PipelineLogic(str, Enum):
    """
    Logic path used for this query (determined by router).
    
    LOGIC_1: Redis cache only (ultra-fast: <10ms)
    LOGIC_2: Pure RAG (no cache, no doc: 1-2s)
    LOGIC_3: Cache + Document (2-4s)
    LOGIC_4: RAG + Document (3-5s)
    """
    LOGIC_1 = "logic_1_redis_only"
    LOGIC_2 = "logic_2_pure_rag"
    LOGIC_3 = "logic_3_redis_with_doc"
    LOGIC_4 = "logic_4_rag_with_doc"


class ResponseStatus(str, Enum):
    """Response status indicator"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"  # Some components failed but partial result available


# ================================================================================
# SECTION 2: USER QUERY SCHEMAS (Request models - from api/request_models.py)
# ================================================================================

class DocumentAttachment(BaseModel):
    """
    File attachment metadata.
    
    Attributes:
        file_name: Original file name (e.g., "document.pdf")
        file_type: File format (pdf, docx, txt)
        file_size: File size in bytes
        file_content: Base64-encoded file content (optional for streaming)
    
    FUTURE EXTENSION (Phase 2):
        - Add file hash (for deduplication)
        - Add upload timestamp
        - Add retention policy
    """
    file_name: str = Field(..., min_length=1, description="Original file name")
    file_type: str = Field(..., pattern="^(pdf|docx|txt)$", description="File format")
    file_size: int = Field(..., gt=0, le=52428800, description="File size in bytes (max 50MB)")
    file_content: Optional[str] = Field(None, description="Base64-encoded file content (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_name": "document.pdf",
                "file_type": "pdf",
                "file_size": 1024000
            }
        }


class UserQueryRequest(BaseModel):
    """
    User query request.
    
    MOVED FROM: api/request_models.py
    
    Attributes:
        user_id: Unique user identifier
        session_id: Session identifier
        prompt: User query/prompt
        document: Optional document attachment
        redis_lookup: Cache lookup flag (YES/NO)
        doc_attached: Document attached flag (YES/NO)
    
    VALIDATION:
        - prompt: non-empty, max 5000 chars
        - user_id: non-empty
        - session_id: non-empty
        - document: optional, only if doc_attached=YES
    
    FUTURE EXTENSION (Phase 2):
        - Add timestamp
        - Add request metadata
        - Add user preferences
    """
    user_id: str = Field(..., min_length=1, description="User ID")
    session_id: str = Field(..., min_length=1, description="Session ID")
    prompt: str = Field(..., min_length=1, max_length=5000, description="User query")
    document: Optional[DocumentAttachment] = Field(None, description="Optional document")
    redis_lookup: RedisLookupFlag = Field(
        RedisLookupFlag.YES,
        description="Check Redis cache?"
    )
    doc_attached: DocAttachedFlag = Field(
        DocAttachedFlag.NO,
        description="Document attached?"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "session_id": "sess_abc123",
                "prompt": "What is machine learning?",
                "redis_lookup": "yes",
                "doc_attached": "no"
            }
        }


# ================================================================================
# SECTION 3: RAG RESULT SCHEMAS (Response models - from api/response_models.py)
# ================================================================================

class SourceAttribution(BaseModel):
    """
    Source metadata for answer attribution.
    
    Attributes:
        source_id: Document/source identifier
        relevance_score: Relevance score (0-1)
        excerpt: Relevant text excerpt from source
    
    FUTURE EXTENSION (Phase 2):
        - Add source URL
        - Add page number
        - Add confidence score
    """
    source_id: str = Field(..., description="Source document ID")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    excerpt: Optional[str] = Field(None, description="Relevant text excerpt")


class RAGResult(BaseModel):
    """
    RAG pipeline result (answer + metadata).
    
    Attributes:
        answer: Generated answer
        sources: List of source attributions
        summary: Document summary (if document attached)
        processing_time_ms: Total processing time
    
    FUTURE EXTENSION (Phase 2):
        - Add confidence score
        - Add alternative answers
        - Add reasoning steps
    """
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceAttribution] = Field(default_factory=list, description="Sources")
    summary: Optional[str] = Field(None, description="Document summary")
    processing_time_ms: int = Field(default=0, description="Processing time in milliseconds")


class RAGResponse(BaseModel):
    """
    User query response.
    
    MOVED FROM: api/response_models.py
    
    Attributes:
        status: Response status (success, error, partial)
        result: RAG result (if success)
        error: Error message (if error)
        logic_path: Which logic path was executed
        request_id: Correlation ID for tracking
    
    FUTURE EXTENSION (Phase 2):
        - Add response timestamp
        - Add metrics (latency breakdown)
        - Add warnings
    """
    status: ResponseStatus = Field(..., description="Response status")
    result: Optional[RAGResult] = Field(None, description="RAG result")
    error: Optional[str] = Field(None, description="Error message")
    logic_path: PipelineLogic = Field(..., description="Logic path executed")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "result": {
                    "answer": "Machine learning is...",
                    "sources": [],
                    "processing_time_ms": 1234
                },
                "logic_path": "logic_2_pure_rag",
                "request_id": "req_abc123"
            }
        }


# ================================================================================
# SECTION 4: SESSION SCHEMAS
# ================================================================================

class SessionMetadata(BaseModel):
    """
    Session metadata.
    
    Attributes:
        user_id: User ID
        session_id: Session ID
        created_at: Session creation timestamp
        last_activity: Last activity timestamp
        query_count: Number of queries in this session
    
    FUTURE EXTENSION (Phase 2):
        - Add device type (mobile, desktop)
        - Add user agent
        - Add IP address (for security)
        - Add session status (active, ended)
    """
    user_id: str
    session_id: str
    created_at: datetime
    last_activity: datetime
    query_count: int = 0


class QueryHistoryItem(BaseModel):
    """
    Single query in history.
    
    Attributes:
        query_id: Unique query identifier
        timestamp: Query timestamp
        prompt: User prompt
        answer: Generated answer
        processing_time_ms: Query processing time
    
    FUTURE EXTENSION (Phase 2):
        - Add feedback score
        - Add user rating
        - Add regenerate count
    """
    query_id: str
    timestamp: datetime
    prompt: str
    answer: str
    processing_time_ms: int


class SessionContext(BaseModel):
    """
    Complete session context.
    
    Attributes:
        metadata: Session metadata
        history: Query history in session
        current_state: Current session state (for continuity)
    
    FUTURE EXTENSION (Phase 2):
        - Add conversation state
        - Add context windows
        - Add learned preferences
    """
    metadata: SessionMetadata
    history: List[QueryHistoryItem] = Field(default_factory=list)
    current_state: Optional[Dict[str, Any]] = None


# ================================================================================
# SECTION 5: ERROR SCHEMAS
# ================================================================================

class ErrorDetail(BaseModel):
    """
    Error detail with context.
    
    Attributes:
        error_code: Machine-readable error code
        error_message: Human-readable error message
        context: Additional context (optional)
    """
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ErrorResponse(BaseModel):
    """
    Error response.
    
    Attributes:
        status: Always "error"
        error: Error details
        request_id: Correlation ID
    
    FUTURE EXTENSION (Phase 2):
        - Add error documentation link
        - Add recovery suggestions
        - Add retry information
    """
    status: ResponseStatus = Field(default=ResponseStatus.ERROR)
    error: ErrorDetail = Field(..., description="Error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
