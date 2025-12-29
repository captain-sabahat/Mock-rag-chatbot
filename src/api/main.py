"""
================================================================================
FILE: src/main.py
================================================================================

PURPOSE:
    FastAPI application factory and initialization. Creates and configures the
    FastAPI app instance, registers all routes, sets up startup/shutdown hooks,
    and initializes all backend components (models, handlers, pipelines).

WORKFLOW:
    1. Load configuration from config/settings.py (at startup, zero runtime I/O)
    2. Initialize FastAPI app instance
    3. Configure CORS (Cross-Origin Resource Sharing) if needed
    4. Register startup hook: initialize models, handlers, pipelines
    5. Register routes from api/routes.py
    6. Register shutdown hook: cleanup resources
    7. Return configured app

IMPORTS:
    - FastAPI: Web framework
    - config.settings: Application settings (env variables, model names, etc.)
    - api.routes: API endpoint definitions
    - utils: Helper functions
    - exceptions: Custom exception handlers
    - container.service_container: CENTRAL ENTRY POINT ✅
    - core handlers: Redis, Vector DB, LLM, SLM handlers

INPUTS:
    - None (uses config/settings.py for configuration)

OUTPUTS:
    - FastAPI app instance (ready to receive HTTP requests)

STARTUP SEQUENCE:
    1. Load Settings (config.settings) - tool names from .env
    2. Initialize ServiceContainer - creates all tools via factories
    3. ServiceContainer reads settings.llm_provider, calls LLMFactory, etc.
    4. All tools created (tool-agnostic)
    5. Register routes
    6. Ready to serve requests

KEY FACTS:
    - Startup happens ONCE when server starts (zero runtime overhead after)
    - All models are cached in memory, never reloaded per request
    - Settings loaded at startup, changes require server restart
    - Error at startup = server fails to start (catches config errors early)
    - Shutdown hook ensures graceful cleanup (close connections, free memory)
    - ServiceContainer is single source of truth for tools
    - No if-else chains for tool selection in main.py
    - Factories handle tool creation (isolated)
    - Tool swap = change .env, restart = done
    - Routes unaware of which tools are active
    - Orchestrator unaware of which tools are active

FUTURE SCOPE (Phase 2+):
    - Add Prometheus metrics endpoint (/metrics)
    - Add health check endpoint (/health) with component status
    - Add request/response logging middleware
    - Add request ID generation (correlation ID tracking)
    - Add error handlers for specific exception types
    - Add structured logging (JSON format)
    - Add graceful shutdown with timeout
    - Add request rate limiting / throttling
    - Add async context manager for database connections
    - Add background task scheduling (APScheduler, Celery)
    - Add OpenTelemetry instrumentation
    - Add environment-specific middleware (dev vs prod)

TESTING ENVIRONMENT:
    - Import: from src.main import app (in tests/conftest.py)
    - Use TestClient: from fastapi.testclient import TestClient
    - Mock models and handlers in tests (use pytest fixtures)
    - Do NOT start real Redis/Vector DB in unit tests (use mocks)

PRODUCTION DEPLOYMENT:
    - Config loaded from environment variables (.env file or system env)
    - Models loaded on first request OR at startup (configurable)
    - All handlers use async/await (non-blocking)
    - Connection pooling for Redis and Vector DB
    - Circuit breaker pattern for failure handling

WORKFLOW:
  1. Load configuration from src/config/settings.py at startup (zero runtime IO)
  2. Initialize FastAPI app instance
  3. Configure CORS if needed
  4. Register startup hook → initialize ServiceContainer (all tools)
  5. Register routes from src/api/routes.py
  6. Register shutdown hook → cleanup resources
  7. Return configured app
"""


from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dotenv import load_dotenv  # ✅ NEW

from src.api import routes
from src.config.settings import Settings
from src.container.service_container import ServiceContainer
from src.core.exceptions import RAGPipelineException
from src.utils import generate_request_id
from src.pipeline.orchestrator import Orchestrator
from src.providers.cache.redis import default_provider as redis_cache_provider  # ✅ NEW

logger = logging.getLogger(__name__)

# Global instances (singleton pattern for startup/shutdown).
_settings: Optional[Settings] = None
_container: Optional[ServiceContainer] = None
_orchestrator: Optional[Orchestrator] = None


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance ready for startup.
    """
    app = FastAPI(
        title="User Chatbot RAG Backend",
        description="Production-grade RAG pipeline for user chatbot",
        version="1.0.0",
    )

    # CORS middleware configuration.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================================================================
    # STARTUP HOOK
    # =========================================================================
    @app.on_event("startup")
    async def startup_event():
        """
        Initialize all components at startup using ServiceContainer.

        SEQUENCE:
        0. Load .env
        1. Load settings from .env
        2. Initialize Redis cache provider
        3. Initialize ServiceContainer (creates all tools via factories)
        4. Verify all tools loaded
        5. Initialize Orchestrator (pass container instance)
        6. Store in global state
        """
        global _settings, _container, _orchestrator

        try:
            logger.info("=" * 80)
            logger.info("APPLICATION STARTUP")
            logger.info("=" * 80)

            # STEP 0: Load .env so Settings sees values
            load_dotenv()  # ✅ NEW: keep default .env path

            # STEP 1: Load settings from .env
            _settings = Settings()
            logger.info(
                "Settings loaded: "
                f"device={_settings.device} | "
                f"llm_provider={_settings.llm_provider} ({_settings.llm_model_name}) | "
                f"slm_provider={_settings.slm_provider} ({_settings.slm_model_name}) | "
                f"embeddings={_settings.embeddings_model_name} | "
                f"vector_db={_settings.vector_db_provider} | "
                f"cache={_settings.cache_provider}"
            )

            # STEP 2: Initialize Redis cache provider (non-blocking if fails)
            logger.info("Initializing Redis cache provider...")
            await redis_cache_provider.initialize()
            if getattr(redis_cache_provider, "initialized", False):
                logger.info("✓ Redis cache initialized and connected")
            else:
                logger.warning("Redis cache not available (running without cache)")

            # STEP 3: Initialize ServiceContainer
            # Pass initialized cache provider into container
            _container = ServiceContainer(_settings, cache_provider=redis_cache_provider)
            await _container.initialize()
            logger.info("✓ ServiceContainer initialized with all tools")

            # STEP 4: Verify all tools loaded and accessible.
            llm = _container.get_llm()
            slm = _container.get_slm()
            embeddings = _container.get_embeddings()
            vector_db = _container.get_vector_db()
            cache = _container.get_cache()
            logger.info("✓ All tools verified and accessible")

            # STEP 5: Initialize Orchestrator
            _orchestrator = Orchestrator(
                container=_container,
                settings=_settings,
            )
            logger.info("✓ Orchestrator initialized")

            logger.info("=" * 80)
            logger.info("APPLICATION STARTUP COMPLETE")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"STARTUP FAILED: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize backend: {str(e)}") from e

    # =========================================================================
    # SHUTDOWN HOOK
    # =========================================================================
    @app.on_event("shutdown")
    async def shutdown_event():
        """
        Cleanup resources at shutdown.

        ACTIONS:
        - Shutdown ServiceContainer (closes all tool connections)
        - Free memory/VRAM
        """
        global _container

        try:
            logger.info("=" * 80)
            logger.info("APPLICATION SHUTDOWN")
            logger.info("=" * 80)

            if _container:
                await _container.shutdown()
                logger.info("✓ ServiceContainer shutdown complete")

            logger.info("=" * 80)
            logger.info("APPLICATION SHUTDOWN COMPLETE")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"SHUTDOWN ERROR: {str(e)}", exc_info=True)

    # =========================================================================
    # EXCEPTION HANDLERS
    # =========================================================================
    @app.exception_handler(RAGPipelineException)
    async def rag_exception_handler(request: Request, exc: RAGPipelineException):
        """Handle custom RAG pipeline exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"RAG error [request_id={request_id}]: {exc.message}",
            extra={"error_code": exc.error_code},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "RAG Pipeline Error",
                "error_code": exc.error_code,
                "message": exc.message,
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"Unexpected error [request_id={request_id}]: {str(exc)}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "request_id": request_id,
            },
        )

    # =========================================================================
    # MIDDLEWARE
    # =========================================================================
    @app.middleware("http")
    async def add_request_id_middleware(request: Request, call_next):
        """Add unique request ID for correlation tracking."""
        request.state.request_id = generate_request_id()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    # =========================================================================
    # ROUTES
    # =========================================================================
    app.include_router(routes.router)

    return app


# Create the app instance when module is imported.
app = create_app()


# =========================================================================
# DEPENDENCY INJECTION HELPERS
# =========================================================================
def get_settings() -> Settings:
    """
    Get global settings instance for dependency injection.
    """
    if _settings is None:
        raise RuntimeError(
            "Settings not initialized. Check application startup logs."
        )
    return _settings


def get_container() -> ServiceContainer:
    """
    Get global ServiceContainer instance for dependency injection.
    """
    if _container is None:
        raise RuntimeError(
            "ServiceContainer not initialized. Check application startup logs."
        )
    return _container


def get_orchestrator() -> Orchestrator:
    """
    Get global Orchestrator instance for dependency injection.
    """
    if _orchestrator is None:
        raise RuntimeError(
            "Orchestrator not initialized. Check application startup logs."
        )
    return _orchestrator
