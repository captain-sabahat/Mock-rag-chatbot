# 4 files: HTTP endpoints
"""
================================================================================
FILE: src/api/__init__.py
================================================================================

PURPOSE:
    Package initialization for API layer. Exports main router for inclusion
    in FastAPI app. Enables clean imports: from src.api import router

WORKFLOW:
    1. Import router from routes.py
    2. Export router as public API
    3. Enable FastAPI app to include router: app.include_router(router)

IMPORTS:
    - APIRouter from fastapi

OUTPUTS:
    - router: FastAPI APIRouter instance (all endpoints)

KEY FACTS:
    - Minimal file (just exports)
    - Enables clean package structure
    - Router contains all HTTP endpoints

FUTURE SCOPE (Phase 2+):
    - Add middleware exports
    - Add dependency exports
    - Add schema exports

TESTING ENVIRONMENT:
    - Import router in tests: from src.api import router

PRODUCTION DEPLOYMENT:
    - FastAPI loads router from this package automatically
"""

# ================================================================================
# IMPORTS
# ================================================================================

from src.api.routes import router

# ================================================================================
# PUBLIC API EXPORTS
# ================================================================================

__all__ = ["router"]
