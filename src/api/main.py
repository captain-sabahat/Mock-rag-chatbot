# ============================================================================
# FastAPI Application Factory
# ============================================================================

"""
FastAPI Application Factory - Create and configure FastAPI app with .env config.

Handles:
- FastAPI app initialization
- Route registration (including new POST monitoring endpoints)
- Middleware setup (CORS, logging, error handling)
- Lifespan management (startup/shutdown)
- Error handling & exception handlers
- OpenAPI documentation configuration
- Monitoring directory initialization
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from pathlib import Path

from .routes import create_api_router
from .models import ErrorResponse
from config import get_settings, get_tool_registry, init_config_loader

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# LIFESPAN CONTEXT MANAGER - Startup & Shutdown Events
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifespan context manager for startup and shutdown events."""
    
    # ════════════════════════════════════════════════════════════════════════
    # STARTUP SEQUENCE
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("🚀 RAG PIPELINE API - STARTING UP")
    logger.info("=" * 80)

    try:
        # Step 1: Initialize configuration system
        logger.info("📂 Initializing configuration system (.env + YAML)...")
        if not init_config_loader():
            raise RuntimeError("Configuration initialization failed")
        logger.info("✅ Configuration loaded successfully")

        # Step 2: Load settings from .env
        logger.info("⚙️ Loading application settings from .env...")
        settings = get_settings()
        logger.info(
            f"✅ Settings loaded from .env\n"
            f" Session Store: {getattr(settings, 'session_store', 'unknown')}\n"
            f" Chunking: {getattr(settings, 'chunking_strategy', 'unknown')}\n"
            f" Embeddings: {getattr(settings, 'embedding_provider', 'unknown')}\n"
            f" Vector DB: {getattr(settings, 'vector_db_provider', 'unknown')}"
        )

        # Step 3: Load tool registry
        logger.info("🔧 Loading tool registry with models from .env...")
        registry = get_tool_registry()
        summary = registry.summary()
        logger.info(
            f"✅ Tool registry loaded with .env models\n"
            f" Active Chunker: {summary['active'].get('chunker', 'N/A')}\n"
            f" Active Embedder: {summary['active'].get('embedder', {}).get('name', 'N/A')}\n"
            f" Active Vector DB: {summary['active'].get('vectordb', {}).get('name', 'N/A')}"
        )

        # Step 4: Initialize monitoring directory
        logger.info("📁 Initializing monitoring directory...")
        monitoring_dir = Path("./data/monitoring")
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Monitoring directory ready: {monitoring_dir}")

        # Step 5: Startup complete
        logger.info("=" * 80)
        logger.info("✅ RAG PIPELINE API - STARTUP COMPLETE")
        logger.info("=" * 80)

                # Step 5: Initialize log buffer for frontend display
        logger.info("📝 Initializing log buffer for live pipeline logs...")
        try:
            from src.core.log_buffer import setup_log_buffer
            
            # Attach buffer handler to all active loggers
            for logger_name in ["src.api.routes", "src.pipeline.orchestrator", "src.core.circuit_breaker"]:
                buffer_logger = logging.getLogger(logger_name)
                setup_log_buffer(buffer_logger)
            
            logger.info("✅ Log buffer initialized - Live logs enabled")
        except Exception as e:
            logger.warning(f"⚠️ Log buffer initialization failed (non-critical): {e}")

    except Exception as e:
        logger.error(
            f"❌ STARTUP FAILED\n"
            f" Error: {str(e)}\n"
            f" Type: {type(e).__name__}",
            exc_info=True
        )
        raise

    yield

    # ════════════════════════════════════════════════════════════════════════
    # SHUTDOWN SEQUENCE
    # ════════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("🛑 RAG PIPELINE API - SHUTTING DOWN")
    logger.info("=" * 80)

    try:
        logger.info("🔄 Cleaning up resources...")
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {str(e)}", exc_info=True)

    logger.info("=" * 80)
    logger.info("✅ RAG PIPELINE API - SHUTDOWN COMPLETE")
    logger.info("=" * 80)

# ============================================================================
# APPLICATION FACTORY
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="RAG Pipeline API",
        description="Document ingestion and retrieval pipeline with monitoring",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ────────────────────────────────────────────────────────────────────────
    # 1. MIDDLEWARE SETUP
    # ────────────────────────────────────────────────────────────────────────

    # CORS middleware - allow frontend requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.debug("✅ CORS middleware configured")

    # ────────────────────────────────────────────────────────────────────────
    # 2. EXCEPTION HANDLERS
    # ────────────────────────────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": str(exc),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "status_code": 500,
            },
        )

    logger.debug("✅ Exception handlers configured")

    # ────────────────────────────────────────────────────────────────────────
    # 3. ROOT ENDPOINT
    # ────────────────────────────────────────────────────────────────────────

    @app.get("/", tags=["Info"])
    async def root():
        """Root endpoint - Returns API information."""
        return {
            "name": "RAG Pipeline API",
            "version": "1.0.0",
            "status": "running",
            "config_source": ".env file",
            "docs": "/docs",
            "redoc": "/redoc",
            "monitoring_dir": "data/monitoring",
            "endpoints": {
                "health": "GET /api/health",
                "ingest_upload": "POST /api/ingest/upload",
                "ingest_all": "GET /api/ingest/all",
                "ingest_status": "GET /api/ingest/status/{ingestion_id}",
                "query": "POST /api/query",
                "monitor_config": "POST /api/monitor/config",
                "monitor_health": "POST /api/monitor/health",
                "monitor_metrics": "POST /api/monitor/metrics",
                "monitor_status": "POST /api/monitor/status",
            },
        }

    logger.debug("✅ Root endpoint configured")

    # ────────────────────────────────────────────────────────────────────────
    # 4. REGISTER API ROUTER
    # ────────────────────────────────────────────────────────────────────────

    router = create_api_router()
    app.include_router(router)

    logger.info("✅ FastAPI application created successfully")
    logger.info(
        f"📖 Documentation available at:\n"
        f" Swagger UI: http://localhost:8000/docs\n"
        f" ReDoc: http://localhost:8000/redoc"
    )

    return app

# ============================================================================
# APP INSTANCE
# ============================================================================

# Create the app instance for uvicorn to import
app = create_app()

# This allows:
# uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
