# ============================================================================
# FASTAPI APPLICATION FACTORY - Create and Configure FastAPI App
# ============================================================================

"""
FastAPI Application Factory - Create and configure FastAPI app with .env config.

Handles:
- FastAPI app initialization
- Route registration
- Middleware setup (CORS, logging, error handling)
- Lifespan management (startup/shutdown)
- Error handling & exception handlers
- OpenAPI documentation configuration
- Load configuration from .env file

ARCHITECTURE:
main.py (THIS FILE) - Creates and configures app
routes.py - Defines API endpoints
orchestrator.py - Coordinates pipeline execution
src/pipeline/ - Executes document processing

DEPENDENCY FLOW:
main.py
├─ Imports config (get_settings, init_config_loader)
├─ Imports tool registry (get_tool_registry)
├─ Imports routes (create_api_router)
├─ Initializes lifespan (startup/shutdown)
└─ Configures middleware & error handlers

NO BUSINESS LOGIC - This file only handles HTTP concerns!
Pipeline execution logic lives in src/pipeline/, not here.

USAGE:
from src.api import create_app

app = create_app()

Run with:
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
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
    """
    Lifespan context manager for startup and shutdown events.
    
    This function runs:
    1. STARTUP CODE - Before app accepts requests
    2. yield - App is running and handling requests
    3. SHUTDOWN CODE - When app is stopping
    
    Startup Tasks:
    - Initialize configuration system (.env + YAML)
    - Load tool registry with models from .env
    - Create session store connections
    - Set up monitoring
    
    Shutdown Tasks:
    - Clean up resources
    - Close database connections
    - Flush logs
    
    Error Handling:
    - Startup failures are fatal (raises exception)
    - Shutdown failures are logged but don't fail
    """

    # ════════════════════════════════════════════════════════════════════════
    # STARTUP SEQUENCE
    # ════════════════════════════════════════════════════════════════════════

    logger.info("=" * 80)
    logger.info("🚀 RAG PIPELINE API - STARTING UP")
    logger.info("=" * 80)

    try:
        # Step 1: Initialize configuration system (loads .env + YAML)
        logger.info("📂 Initializing configuration system (.env + YAML)...")
        if not init_config_loader():
            raise RuntimeError("Configuration initialization failed")
        logger.info("✅ Configuration loaded successfully")

        # Step 2: Load settings from .env
        logger.info("⚙️ Loading application settings from .env...")
        settings = get_settings()
        logger.info(
            f"✅ Settings loaded from .env\n"
            f" Session Store: {settings.session_store.backend}\n"
            f" Chunking: {settings.chunking.strategy}\n"
            f" Embeddings: {settings.embeddings.active_provider}\n"
            f" Vector DB: {settings.vectordb.active_provider}"
        )

        # Step 3: Load tool registry (reads tool names + models from .env)
        logger.info("🔧 Loading tool registry with models from .env...")
        registry = get_tool_registry()

        # Get registry summary (includes .env models)
        summary = registry.summary()
        logger.info(
            f"✅ Tool registry loaded with .env models\n"
            f" Active Chunker: {summary['active']['chunker']}\n"
            f" Active Embedder: {summary['active']['embedder']['name']} "
            f"(model: {summary['active']['embedder'].get('model', 'N/A')})\n"
            f" Active Vector DB: {summary['active']['vectordb']['name']}"
        )

        # Log available tools for debugging
        logger.debug(
            f"📋 Available tools:\n"
            f" Chunkers: {summary['available']['chunkers']}\n"
            f" Embedders: {summary['available']['embedders']}\n"
            f" Vector DBs: {summary['available']['vectordbs']}"
        )

        # Step 4: Startup complete
        logger.info("=" * 80)
        logger.info("✅ RAG PIPELINE API - STARTUP COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(
            f"❌ STARTUP FAILED\n"
            f" Error: {str(e)}\n"
            f" Type: {type(e).__name__}",
            exc_info=True
        )
        raise

    # ════════════════════════════════════════════════════════════════════════
    # APP IS RUNNING...
    # ════════════════════════════════════════════════════════════════════════

    yield  # App is now running

    # ════════════════════════════════════════════════════════════════════════
    # SHUTDOWN SEQUENCE
    # ════════════════════════════════════════════════════════════════════════

    logger.info("=" * 80)
    logger.info("🛑 RAG PIPELINE API - SHUTTING DOWN")
    logger.info("=" * 80)

    try:
        # Step 1: Close session store connections
        logger.info("🔌 Closing session store connections...")
        try:
            from src.cache.session_store_factory import SessionStoreFactory
            store = SessionStoreFactory.create_store(settings)
            if hasattr(store, 'close'):
                await store.close()
            logger.info("✅ Session store closed")
        except Exception as e:
            logger.warning(f"⚠️ Could not close session store: {str(e)}")

        # Step 2: Final log
        logger.info("=" * 80)
        logger.info("✅ RAG PIPELINE API - SHUTDOWN COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(
            f"⚠️ Error during shutdown: {str(e)}",
            exc_info=True
        )
        # Don't raise - shutdown errors shouldn't crash the app

# ============================================================================
# FASTAPI APPLICATION FACTORY
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Performs:
    1. Initialize FastAPI with lifespan management
    2. Add CORS middleware
    3. Add exception handlers
    4. Include API routes
    5. Configure OpenAPI documentation
    
    Returns:
        Configured FastAPI application instance
    
    Example:
        >>> app = create_app()
        >>> # Run with: uvicorn src.api.main:app --reload
    """

    logger.info("📦 Creating FastAPI application instance...")

    # ────────────────────────────────────────────────────────────────────────
    # 1. CREATE APP WITH LIFESPAN (loads .env config during startup)
    # ────────────────────────────────────────────────────────────────────────

    app = FastAPI(
        title="RAG Pipeline API",
        description="Retrieval Augmented Generation (RAG) Pipeline - "
                    "Upload documents, ask questions, get answers. "
                    "Configuration loaded from .env file.",
        version="1.0.0",
        lifespan=lifespan,  # Attach startup/shutdown handlers
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    logger.debug("✅ FastAPI app instance created")

    # ────────────────────────────────────────────────────────────────────────
    # 2. ADD CORS MIDDLEWARE
    # ────────────────────────────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        # ⚠️ TODO PRODUCTION: Restrict to specific frontend domains
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.debug("✅ CORS middleware configured")

    # ────────────────────────────────────────────────────────────────────────
    # 3. ADD EXCEPTION HANDLERS
    # ────────────────────────────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch uncaught exceptions and return structured JSON error response.
        
        Example response:
        {
            "error_code": "internal_server_error",
            "message": "An unexpected error occurred",
            "details": "Division by zero"
        }
        """
        logger.error(
            f"❌ Unhandled exception in {request.url.path}",
            exc_info=exc
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="internal_server_error",
                message="An unexpected error occurred",
                details=str(exc)
            ).model_dump()
        )

    logger.debug("✅ Exception handlers configured")

    # ────────────────────────────────────────────────────────────────────────
    # 4. INCLUDE API ROUTES
    # ────────────────────────────────────────────────────────────────────────

    api_router = create_api_router()
    app.include_router(api_router)

    logger.debug("✅ API routes included")

    # ────────────────────────────────────────────────────────────────────────
    # 5. ROOT ENDPOINT
    # ────────────────────────────────────────────────────────────────────────

    @app.get("/", tags=["Info"])
    async def root():
        """
        Root endpoint - Returns API information and available endpoints.
        
        Useful for:
        - Checking API is running
        - Discovering endpoints
        - Getting API version
        """
        return {
            "name": "RAG Pipeline API",
            "version": "1.0.0",
            "status": "running",
            "config_source": ".env file",
            "docs": "/docs",
            "redoc": "/redoc",
            "endpoints": {
                "health": "GET /api/health",
                "upload": "POST /api/upload",
                "query": "POST /api/query",
                "status": "GET /api/status/{request_id}",
                "tools": "GET /api/tools",  # New: shows tools from .env
            },
        }

    logger.debug("✅ Root endpoint configured")

    # ────────────────────────────────────────────────────────────────────────
    # 6. TOOLS ENDPOINT (shows .env configuration)
    # ────────────────────────────────────────────────────────────────────────

    @app.get("/api/tools", tags=["Info"])
    async def get_tools_info():
        """
        Get information about available tools and active tools from .env.
        
        Returns tool names and model identifiers loaded from .env file.
        """
        try:
            registry = get_tool_registry()
            return registry.summary()
        except Exception as e:
            logger.error(f"Error getting tool registry: {str(e)}", exc_info=True)
            return {
                "error": "Failed to load tool registry",
                "details": str(e)
            }

    logger.debug("✅ Tools endpoint configured")

    # ────────────────────────────────────────────────────────────────────────
    # APPLICATION READY...
    # ────────────────────────────────────────────────────────────────────────

    logger.info("✅ FastAPI application created successfully")
    logger.info(
        f"📖 Documentation available at:\n"
        f" Swagger UI: http://localhost:8000/docs\n"
        f" ReDoc: http://localhost:8000/redoc\n"
        f" Tools Info: http://localhost:8000/api/tools"
    )

    return app

# ============================================================================
# APP INSTANCE
# ============================================================================

# Create the app instance for uvicorn to import
app = create_app()

# This allows:
# uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000