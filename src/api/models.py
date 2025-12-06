"""
================================================================================
API MODELS - Pydantic Request/Response Schemas
================================================================================

PURPOSE:
--------
Define all HTTP request/response models for the RAG pipeline API.

These models:
  - Validate incoming requests
  - Serialize pipeline outputs
  - Generate OpenAPI documentation
  - Provide type hints to frontend

ARCHITECTURE:
--------------
API → models.py (validation) → routes.py (http handlers)
                                    ↓
                           orchestrator.py (pipeline)
                                    ↓
                           cache/ + tools/

NO BUSINESS LOGIC: Models only define schemas, no pipeline logic here.

================================================================================
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# HEALTH CHECK
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field("1.0.0", description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-11-29T13:21:00Z",
                "version": "1.0.0"
            }
        }


# ============================================================================
# UPLOAD DOCUMENT
# ============================================================================

class UploadRequest(BaseModel):
    """Request to upload a document."""
    file_name: str = Field(..., description="Original file name (PDF, TXT, JSON)")
    file_content: bytes = Field(..., description="Raw file bytes")
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="Optional metadata (author, source, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_name": "document.pdf",
                "metadata": {"source": "user_upload", "category": "research"}
            }
        }


class UploadResponse(BaseModel):
    """Response after document upload."""
    request_id: str = Field(..., description="Unique session ID for tracking")
    file_name: str = Field(..., description="Uploaded file name")
    status: str = Field("processing", description="Current status (processing/completed/error)")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_1234567890",
                "file_name": "document.pdf",
                "status": "processing",
                "message": "Document uploaded and ingestion started",
                "timestamp": "2025-11-29T13:21:00Z"
            }
        }


# ============================================================================
# QUERY / SEARCH
# ============================================================================

class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    query: str = Field(..., min_length=1, description="User query/question")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    session_id: Optional[str] = Field(None, description="Session ID for context (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the benefits of RAG systems?",
                "top_k": 5
            }
        }


class QueryResult(BaseModel):
    """Single search result."""
    rank: int = Field(..., description="Result rank (1-based)")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class QueryResponse(BaseModel):
    """Response with search results."""
    request_id: str = Field(..., description="Session/request ID")
    query: str = Field(..., description="Original query")
    results: List[QueryResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    processing_time_ms: float = Field(..., description="Query processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_1234567890",
                "query": "What are the benefits of RAG systems?",
                "results": [
                    {
                        "rank": 1,
                        "content": "RAG systems improve accuracy by...",
                        "score": 0.95,
                        "metadata": {"doc": "document.pdf", "chunk": 1}
                    }
                ],
                "total_results": 1,
                "processing_time_ms": 45.3,
                "timestamp": "2025-11-29T13:21:00Z"
            }
        }


# ============================================================================
# SESSION STATUS
# ============================================================================

class NodeStatus(BaseModel):
    """Status of a single pipeline node."""
    node_name: str = Field(..., description="Node name (ingestion, preprocessing, etc.)")
    status: str = Field(..., description="Status (pending/running/completed/failed)")
    input_valid: bool = Field(True, description="Input validation passed?")
    output_ready: bool = Field(False, description="Output ready for next node?")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionStatusResponse(BaseModel):
    """Response with session/pipeline status."""
    request_id: str = Field(..., description="Session ID")
    file_name: str = Field(..., description="Original file name")
    overall_status: str = Field(..., description="Overall status (processing/completed/error)")
    nodes: List[NodeStatus] = Field(..., description="Status of each pipeline node")
    progress_percent: int = Field(0, ge=0, le=100, description="Progress percentage")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_1234567890",
                "file_name": "document.pdf",
                "overall_status": "processing",
                "nodes": [
                    {
                        "node_name": "ingestion",
                        "status": "completed",
                        "input_valid": True,
                        "output_ready": True
                    },
                    {
                        "node_name": "preprocessing",
                        "status": "running",
                        "input_valid": True,
                        "output_ready": False
                    }
                ],
                "progress_percent": 40
            }
        }


# ============================================================================
# ERROR RESPONSE
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    error_code: str = Field(..., description="Error code (validation_error, processing_error, etc.)")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "validation_error",
                "message": "Invalid file format",
                "details": {"expected": "PDF/TXT/JSON", "received": "DOCX"},
                "timestamp": "2025-11-29T13:21:00Z"
            }
        }
