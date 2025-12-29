"""
================================================================================
FILE: src/pipelines/base.py
================================================================================

PURPOSE:
    Abstract base class for all pipeline implementations. Defines interface
    that all line orchestrators (LOGIC_1/2/3/4) must implement.

WORKFLOW:
    1. Define abstract methods (execute, validate_input, etc.)
    2. Each pipeline inherits and implements these methods
    3. Enables polymorphism: call any pipeline with same interface
    4. Common utilities for all pipelines

IMPORTS:
    - ABC, abstractmethod: Abstract base class
    - asyncio: Async operations
    - logging: Logging
    - config: Settings
    - schemas: Request/response models

INPUTS:
    - request: UserQueryRequest (prompt, document, flags)
    - handlers: Dict of core handlers
    - models: Dict of initialized models
    - settings: Configuration

OUTPUTS:
    - RAGResponse (status, result, logic_path)

PIPELINE INTERFACE:
    - async execute(request) → RAGResponse
    - async validate_input(request) → bool
    - async log_execution(request, response) → None

KEY FACTS:
    - All methods are async (non-blocking)
    - Base class provides common logging/error handling
    - Each line implements specific logic
    - Enables easy testing (mock base pipeline)

PIPELINE TYPES (Implementations):
    - LOGIC_1: Redis cache only (<10ms)
    - LOGIC_2: Pure RAG (1-2s)
    - LOGIC_3: Cache + Document (3-4s)
    - LOGIC_4: RAG + Document (4-5s)

FUTURE SCOPE (Phase 2+):
    - Add pipeline metrics collection
    - Add pipeline state machine
    - Add pipeline composition
    - Add conditional logic
    - Add branching pipelines
    - Add dynamic pipeline selection

TESTING ENVIRONMENT:
    - Create mock pipelines inheriting from BasePipeline
    - Test execute() with various inputs
    - Verify error handling
    - Test logging

PRODUCTION DEPLOYMENT:
    - All pipelines inherit and implement BasePipeline
    - Single interface for all logic paths
    - Easy to add new pipelines

IMPORTS:

- ABC, abstractmethod: Abstract base class
- asyncio: Async operations
- logging: Logging
- config: Settings
- schemas: Request/response models
"""

# ================================================================================
# IMPORTS
# ================================================================================

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from .schemas import UserQueryRequest, RAGResponse, ResponseStatus, PipelineLogic
from src.config.settings import Settings
from src.container.service_container import ServiceContainer

logger = logging.getLogger(__name__)

# ================================================================================
# BASE PIPELINE CLASS
# ================================================================================


class BasePipeline(ABC):
    """
    Abstract base class for all pipeline implementations.

    New contract (after refactor):
      - Pipelines receive Settings + ServiceContainer
      - Tools (llm, vectordb, cache, etc.) are fetched from the container
        inside concrete pipelines, not passed as dicts.
    """

    def __init__(
        self,
        settings: Settings,
        container: ServiceContainer,
    ) -> None:
        """
        Initialize base pipeline.

        Args:
            settings: Application settings.
            container: ServiceContainer exposing all providers.
        """
        self.settings = settings
        self.container = container

    # --------------------------------------------------------------------------
    # Abstract interface
    # --------------------------------------------------------------------------

    @abstractmethod
    async def execute(self, request: UserQueryRequest) -> RAGResponse:
        """
        Execute pipeline logic.

        Must be implemented by subclasses.

        Args:
            request: User query request.

        Returns:
            RAGResponse with result or error.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------------
    # Common helpers
    # --------------------------------------------------------------------------

    async def validate_input(self, request: UserQueryRequest) -> bool:
        """
        Validate input before execution.

        Args:
            request: User query request.

        Returns:
            True if valid, False otherwise.
        """
        if not request.prompt or len(request.prompt) == 0:
            return False
        if len(request.prompt) > 5000:
            return False
        return True

    async def log_execution(
        self,
        request: UserQueryRequest,
        response: RAGResponse,
        processing_time_ms: int,
    ) -> None:
        """
        Log pipeline execution.

        Args:
            request: User query request.
            response: Pipeline response.
            processing_time_ms: Execution time in milliseconds.
        """
        logger.info(
            f"Pipeline {response.logic_path} executed",
            extra={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "status": response.status,
                "processing_time_ms": processing_time_ms,
                "logic_path": response.logic_path,
            },
        )

    async def handle_error(
        self,
        request: UserQueryRequest,
        error: Exception,
        logic_path: PipelineLogic,
    ) -> RAGResponse:
        """
        Handle errors during execution.

        Args:
            request: Original request.
            error: Exception that occurred.
            logic_path: Which logic path was executing.

        Returns:
            Error RAGResponse.
        """
        error_message = str(error)
        logger.error(
            f"Pipeline error in {logic_path}: {error_message}",
            extra={
                "user_id": request.user_id,
                "session_id": request.session_id,
            },
            exc_info=True,
        )

        return RAGResponse(
            status=ResponseStatus.ERROR,
            error=error_message,
            logic_path=logic_path,
        )


# ================================================================================
# FUTURE EXTENSION (Phase 2+)
# ================================================================================

# TODO (Phase 2): Add pipeline metrics collection
# TODO (Phase 2): Add pipeline state machine
# TODO (Phase 2): Add pipeline composition
# TODO (Phase 2): Add conditional logic/branching
# TODO (Phase 2): Add dynamic pipeline selection
