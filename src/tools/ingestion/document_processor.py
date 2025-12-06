"""
===============================================================================
document_processor.py
===============================================================================

SUMMARY:
--------
This module implements the core orchestration class for document ingestion,
routing, and processing within the pipeline. It manages multi-format document
routing (PDF, Text, JSON, Markdown), parser lifecycle, metrics collection, error
handling, and circuit breaker integration for resilient operation.

WORKING & METHODOLOGY:
----------------------
The DocumentProcessor class holds configuration parameters and manages the
overall document processing flow. The main method `process_document()` accepts
raw input data (bytes), determines content type, selects appropriate parser,
executes parsing, validates, and routes documents through various stages.

It carefully manages exceptions, updates processing state, tracks metrics,
and invokes circuit breakers if needed to avoid cascading failures. It also
generates detailed logs and optional MLOps metrics for monitoring.

INPUTS:
-------
- request_id (str): Unique request identifier
- file_name (str): Name of the uploaded file
- file_content (bytes): Raw file bytes for parsing
- config (dict): Optional configuration override

OUTPUTS:
--------
- PipelineState object: Contains current state, metrics, errors, progress
- Updates stored via the SessionStore (external dependency)

GLOBAL & CONFIG VARIABLES:
---------------------------
- self.settings: Loaded from configuration, includes pipeline parameters
- self.session_store: External session storage handler
- self.parsers_map: Dictionary mapping content types to parser classes
- self.metrics_collector: Placeholder for MLOps metrics (can be uncommented)

FUTURE WORK:
------------
- Integrate MLFlow or other monitoring to track metrics/artifacts
- Support additional formats and parsers
- Enhance error recovery and retries
- Expand circuit breaker strategies

CIRCUIT_BREAK:
--------------
Critical sections include parsing and routing stages. Failures raise exceptions
that may trip circuit breakers, halting subsequent processing to avoid cascade.

MONITORING & HEALTH:
--------------------
Use MLOps hooks in critical steps for runtime resource monitoring, performance,
and error tracking, enabling post-mortem analysis and alerts.

===============================================================================
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from pydantic import ValidationError

# Import necessary parsers
from .models import DocumentInput, ProcessingResult
from .parsers import base_parser, pdf_parser, text_parser, json_parser, markdown_parser

# Placeholder for circuit breaker logic
from src.core.circuit_breaker import CircuitBreakerManager

class DocumentProcessor:
    """
    Core orchestrator for document ingestion and processing pipeline.

    Responsible for:
    - Routing files based on content type
    - Managing parser lifecycle
    - Collecting metrics (MLOps)
    - Error handling & rollback
    - Circuit breaker integration
    """

    def __init__(self, settings: Dict[str, Any], session_store: Any):
        """
        Initialize with configuration settings and session store handler.
        """
        self.settings = settings
        self.session_store = session_store

        # Map content types to parser classes
        self.parsers_map = {
            "application/pdf": pdf_parser.PDFParser,
            "text/plain": text_parser.TextParser,
            "application/json": json_parser.JSONParser,
            "text/markdown": markdown_parser.MarkdownParser,
        }

        # Placeholder for circuit breaker manager
        from src.core.circuit_breaker import CircuitBreakerManager
        self.circuit_breaker_manager = CircuitBreakerManager()

        # Logger setup
        self.logger = logging.getLogger("DocumentProcessor")
        self.logger.info(f"Initialized DocumentProcessor with settings: {self.settings}")

    async def process_document(self, request_id: str, file_name: str, file_content: bytes) -> ProcessingResult:
        """
        Main entry: accepts raw bytes, processes document with routing, parsing,
        validation, metrics, error handling, and status updates.

        Returns:
            ProcessingResult: Current state with parsed data, metrics, errors.
        """
        # Initialize state
        state = DocumentInput(
            request_id=request_id,
            file_name=file_name,
            raw_content=file_content,
            status="processing",
            progress_percent=0,
            checkpoints={},
            messages=[],
            errors=[]
        )

        # Update session store: start
        await self.session_store.set_session(state)

        # Detect content type
        content_type = self._detect_content_type(file_content, file_name)

        # Get parser class
        parser_class = self.parsers_map.get(content_type)
        if parser_class is None:
            error_msg = f"Unsupported content type: {content_type}"
            self._log_error(state, error_msg)
            state.errors.append(error_msg)
            state.status = "error"
            await self.session_store.set_session(state)
            return state

        # Instantiate parser
        parser = parser_class()

        # Check circuit breaker before parsing
        breaker = self.circuit_breaker_manager.get_breaker(request_id)
        if not breaker.can_attempt():
            error_msg = f"Circuit breaker open, aborting processing for {request_id}"
            self._log_error(state, error_msg)
            state.errors.append(error_msg)
            state.status = "error"
            await self.session_store.set_session(state)
            return state

        try:
            # Execute parsing with metrics logging
            # #MLFLOW:METRIC:ParsingStartTime
            parsed_text = await self._execute_with_metrics(parser.parse, file_content)
            # #MLFLOW:METRIC:ParsingEndTime

            # Update state with parsed content
            state.parsed_text = parsed_text
            state.progress_percent = 20
            await self.session_store.set_session(state)

            # Example validation step (can be extended)
            # Validate parsed_text content
            if not parsed_text or len(parsed_text.strip()) == 0:
                raise ValueError("Parsed text is empty.")

            # Proceed to further processing...
            # e.g., chunking, embedding, storage (not implemented here)

        except Exception as e:
            # Log error, record checkpoint, possibly trip circuit breaker
            self._log_error(state, str(e))
            state.errors.append(str(e))
            state.status = "error"
            # #MLFLOW:METRIC:ErrorOccurred
            # Circuit breaker recording failure
            breaker.record_failure()
            await self.session_store.set_session(state)
            return state

        # Finalize processing (simulate step completion)
        state.progress_percent = 100
        state.status = "completed"
        await self.session_store.set_session(state)

        # Return final state
        return state

    def _detect_content_type(self, content: bytes, filename: str) -> str:
        """
        Detect file content type based on bytes and filename extension.
        Can be extended with real mime type detection.
        """
        # Basic extension-based detection
        extension = filename.split('.')[-1].lower()
        if extension == "pdf":
            return "application/pdf"
        elif extension == "txt":
            return "text/plain"
        elif extension == "json":
            return "application/json"
        elif extension in ["md", "markdown"]:
            return "text/markdown"
        # Else, default to plain text for now
        return "text/plain"

    async def _execute_with_metrics(self, func, *args, **kwargs):
        """
        Execute parser function with metrics collection.
        Placeholder for MLOps integration (uncomment for logging).
        """
        # start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        # duration_ms = (time.perf_counter() - start_time) * 1000
        # #MLFLOW:METRIC:ParsingDurationMS = duration_ms
        # #MLFLOW:METRIC:InputSizeBytes = len(args[0])
        # #MLFLOW:METRIC:OutputSizeBytes = len(result.encode('utf-8'))
        return result

    def _log_error(self, state: DocumentInput, message: str):
        """
        Log errors with minimal info.
        """
        self.logger.error(f"Error processing {state.request_id}: {message}")
        # Optionally, extend with external alerting

# Note for future contributors: To extend parsing techniques or formats,
# add new parser classes to `parsers/` and update `self.parsers_map`
# accordingly in __init__.py for modularity.

# This class can lead to circuit break situations if parse() fails repeatedly.
# The circuit breaker monitors failure rate and during high failures,
# it opens, preventing further processing to protect downstream components.

# Metrics to log (for MLFlow or custom tools):
# - Parsing start/end time
# - Input bytes size
# - Output text size
# - Errors count
# Monitoring resource usage in _execute_with_metrics or via external tools.
