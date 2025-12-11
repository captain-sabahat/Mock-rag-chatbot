# ============================================================================
# API Models - Request and Response Schemas
# ============================================================================

"""
Pydantic models for API request/response validation.
Used for type hints and OpenAPI documentation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# REQUEST MODELS
# ============================================================================


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., min_length=2, description="Search query")
    topk: Optional[int] = Field(5, ge=1, le=100, description="Number of results to return")
    session_id: Optional[str] = Field(None, description="Session ID for contextual queries")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "topk": 5,
                "session_id": "sess_123",
            }
        }


# ============================================================================
# QUERY RESPONSE MODELS
# ============================================================================


class QueryResult(BaseModel):
    """Single query result."""
    content: str
    score: float
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    results: List[QueryResult] = []

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "content": "Machine learning is a field of AI that focuses on learning from data.",
                        "score": 0.95,
                        "source": "document_1.pdf",
                        "metadata": {"page": 3, "chunk_id": 12},
                    }
                ]
            }
        }


# ============================================================================
# GENERIC ERROR RESPONSE
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    details: Optional[str] = None
    timestamp: str
    status_code: int

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Configuration error",
                "details": "vector_db_provider not found",
                "timestamp": "2025-12-10T21:00:00Z",
                "status_code": 500,
            }
        }


# ============================================================================
# MONITORING / HEALTH RESPONSES
# ============================================================================


class MonitoringStatusResponse(BaseModel):
    """Response for POST /api/monitor/status endpoint."""
    timestamp: str
    backend_connected: bool
    status: str
    config: Dict[str, Any] = {}
    health: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    data_files: Dict[str, str] = {}


class HealthCheckResponse(BaseModel):
    """Response for POST /api/monitor/health endpoint."""
    timestamp: str
    status: str
    backend_connected: bool
    configuration: Dict[str, Any] = {}
    circuit_breaker: Dict[str, Any] = {}
    nodes: Dict[str, Any] = {}
    pipeline: Dict[str, Any] = {}
    orchestrator: Dict[str, Any] = {}


class ConfigResponse(BaseModel):
    """Response for POST /api/monitor/config endpoint."""
    timestamp: str
    status: str
    backend_connected: bool
    configuration: Dict[str, Any] = {}


class MetricsResponse(BaseModel):
    """Response for POST /api/monitor/metrics endpoint."""
    timestamp: str
    summary: Dict[str, int] = {}
    ingestions: List[Dict[str, Any]] = []
