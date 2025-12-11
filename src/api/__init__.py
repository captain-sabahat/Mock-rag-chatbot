"""
================================================================================
RAG PIPELINE - API PACKAGE INITIALIZER
================================================================================

PURPOSE:
--------
FastAPI HTTP layer for the RAG pipeline.

Exports:
  - FastAPI app factory (create_app)
  - Pydantic request/response models
  - API router

ARCHITECTURE LAYER:
-------------------
     User (HTTP Request)
           ↓
     src/api/  ← This package
           ↓
     src/pipeline/  ← Orchestration
           ↓
     src/cache/ + src/tools/ ← Implementations
           ↓
     config/  ← Settings + Registry

NO REVERSE IMPORTS: This layer never imports src/tools or src/pipeline directly.
Instead, it calls src/pipeline.orchestrator which handles everything.

USAGE EXAMPLE:
--------------
    from src.api import create_app
    
    app = create_app()
    # Run with: uvicorn src.api.main:app --reload

================================================================================
"""

# Models (Pydantic request/response schemas)
from .models import (
    QueryRequest,
    QueryResponse,
    ErrorResponse,
    MonitoringStatusResponse,
    HealthCheckResponse,
    ConfigResponse,
    MetricsResponse,
)

# Router (FastAPI routes)
from .routes import create_api_router

# App factory (FastAPI application)
from .main import create_app

__all__ = [
    # App
    "create_app",
    "create_api_router",
    # Models
    "HealthResponse",
    "UploadRequest",
    "UploadResponse",
    "QueryRequest",
    "QueryResponse",
    "SessionStatusResponse",
    "ErrorResponse",
]