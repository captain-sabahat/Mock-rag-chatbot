# ASGI entry point
"""
================================================================================
FILE: src/asgi.py
================================================================================

PURPOSE:
    ASGI (Asynchronous Server Gateway Interface) entry point for production servers
    (Gunicorn, Uvicorn, AWS Lambda, etc.). This is the standard interface that
    production ASGI servers expect to start the FastAPI application.

WORKFLOW:
    1. Import FastAPI application from main.py (already initialized)
    2. Expose the 'app' object for ASGI servers to call
    3. Handle any pre-server hooks or configurations
    4. Return the ASGI callable

IMPORTS:
    - FastAPI app instance from main.py

INPUTS:
    - None (ASGI servers handle input routing)

OUTPUTS:
    - ASGI application object (callable) that responds to ASGI interface

KEY FACTS:
    - This file MUST exist for production deployment
    - Used by: gunicorn, uvicorn, AWS Lambda, Heroku, Docker containers
    - Never modify app behavior in this file; use main.py for app setup
    - This is the single entry point for all production server types

FUTURE SCOPE (Phase 2+):
    - Add ASGI middleware for request/response timing
    - Add environment-specific configuration loading
    - Add pre-flight health checks before accepting requests
    - Add graceful shutdown handlers
    - Add production server logging hooks
    - Support for multiple worker processes coordination

TESTING ENVIRONMENT:
    - Set PYTHONPATH=. before running tests
    - Use hypercorn or uvicorn in test command: uvicorn src.asgi:app --reload
"""

# ================================================================================
# IMPORTS
# ================================================================================

from .main import app
from fastapi.middleware.cors import CORSMiddleware

# ================================================================================
# ASGI APPLICATION EXPORT
# ================================================================================

# This is the ASGI callable that production servers expect
# Do NOT rename this variable; ASGI servers look for 'app' by default
__all__ = ["app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
