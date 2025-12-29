# Main coordinator
"""
================================================================================
FILE: src/pipelines/orchestrator.py
================================================================================

PURPOSE:
    Main orchestrator that coordinates all pipeline execution.
    Routes requests through logic_router to determine which pipeline to use.
    Manages execution of LOGIC_1/2/3/4 pipelines.

WORKFLOW:
    1. Receive UserQueryRequest
    2. Get tools from ServiceContainer (llm, slm, embeddings, vectordb, cache)
    2. Call logic_router to determine which pipeline (LOGIC_1/2/3/4)
    3. If LOGIC_1 (cache HIT): return cached result
    4. If LOGIC_2 (pure RAG): run RAG pipeline
    5. If LOGIC_3 (cache+doc): run doc summary + RAG
    6. If LOGIC_4 (RAG+doc): run doc summary + RAG
    7. Return RAGResponse to caller

LATENCY TARGETS:
    - LOGIC_1 (cache HIT): <10ms
    - LOGIC_2 (pure RAG): <2s
    - LOGIC_3 (cache+doc): <3s
    - LOGIC_4 (RAG+doc): <4s

IMPORTS:
    - asyncio: Async operations
    - logging: Logging
    - pipelines: All pipeline implementations
    - logic_router: LogicRouter for routing
    - core handlers: All backend handlers

INPUTS:
    - request: UserQueryRequest
    - handlers: Dict of all handlers
    - models: Dict of all models
    - settings: Application settings

OUTPUTS:
    - RAGResponse (status, result, logic_path, request_id)

ORCHESTRATOR RESPONSIBILITIES:
    1. Route to correct pipeline
    2. Execute pipeline
    3. Handle errors gracefully
    4. Track metrics (latency, success rate)
    5. Cache results

KEY FACTS:
    - Central coordinator for all logic paths
    - Enables flexible routing
    - Collects metrics
    - Handles errors at top level

FUTURE SCOPE (Phase 2+):
    - Add observability (OpenTelemetry)
    - Add metrics collection (Prometheus)
    - Add distributed tracing
    - Add fallback strategies
    - Add A/B testing

TESTING ENVIRONMENT:
    - Mock all handlers
    - Test all logic paths
    - Verify routing logic
    - Test error handling

PRODUCTION DEPLOYMENT:
    - Monitor latency per logic path
    - Alert on errors
    - Track success rate
    - Implement graceful degradation
"""

# FILE: src/pipeline/orchestrator.py

# Main coordinator
"""
================================================================================
FILE: src/pipelines/orchestrator.py
================================================================================

PURPOSE:
    Main orchestrator that coordinates all pipeline execution.
    Routes requests through logic_router to determine which pipeline to use.
    Manages execution of LOGIC_1/2/3/4 pipelines.
"""

import asyncio
import logging
import time
import datetime
from typing import Any, Optional

from .schemas import UserQueryRequest, RAGResponse, ResponseStatus, PipelineLogic
from src.config.settings import Settings
from src.container.service_container import ServiceContainer
from src.pipeline.logic_router import LogicRouter
from src.pipeline.redis_cache import RedisCachePipeline
from src.pipeline.rag import RAGPipeline
from src.core.exceptions import RAGPipelineException

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator for RAG pipeline execution (tool-agnostic).
    """

    def __init__(
        self,
        container: ServiceContainer,
        settings: Settings,
    ) -> None:
        """
        Initialize orchestrator with tool-agnostic ServiceContainer.

        Args:
            container: ServiceContainer with all tools (factory-created)
            settings: Application settings
        """
        self.container = container
        self.settings = settings

        # Initialize routing + pipelines
        self.logic_router = LogicRouter(settings=self.settings)

        # ✅ CORRECT ORDER: settings FIRST, then container
        self.cache_pipeline = RedisCachePipeline(settings, container)
        self.rag_pipeline = RAGPipeline(settings, container)

        logger.info("✓ Orchestrator initialized (tool-agnostic)")

    async def execute(self, request: UserQueryRequest) -> RAGResponse:
        """
        Execute appropriate pipeline based on request.

        Args:
            request: User query request

        Returns:
            RAGResponse with result or error
        """
        start_time = time.time()

        try:
            logger.info(
                f"Orchestrator executing for user {request.user_id}",
                extra={
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "prompt_length": len(request.prompt),
                },
            )

            # ============================================================
            # STEP 1: Route request → which pipeline to use
            # ============================================================

            logic_path, cached_result, reason = await self.logic_router.route(
                request
            )

            logger.debug(f"Routing decision: {logic_path} ({reason})")

            # ============================================================
            # STEP 2: Execute appropriate pipeline
            # ============================================================

            if logic_path == PipelineLogic.LOGIC_1:
                # Cache HIT - return cached result
                response = await self.cache_pipeline.execute(
                    request, cached_result
                )

            elif logic_path == PipelineLogic.LOGIC_2:
                # Pure RAG (no cache, no docs)
                response = await self.rag_pipeline.execute(
                    request, PipelineLogic.LOGIC_2
                )

            elif logic_path == PipelineLogic.LOGIC_3:
                # Cache + Document processing
                response = await self.rag_pipeline.execute(
                    request, PipelineLogic.LOGIC_3
                )

            elif logic_path == PipelineLogic.LOGIC_4:
                # RAG + Document processing
                response = await self.rag_pipeline.execute(
                    request, PipelineLogic.LOGIC_4
                )

            else:
                # Unknown logic path (should not happen)
                response = RAGResponse(
                    status=ResponseStatus.ERROR,
                    error=f"Unknown logic path: {logic_path}",
                    logic_path=PipelineLogic.LOGIC_2,
                )

            # ============================================================
            # STEP 3: Add processing metrics
            # ============================================================

            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Orchestrator completed ({logic_path})",
                extra={
                    "user_id": request.user_id,
                    "logic_path": response.logic_path,
                    "status": response.status,
                    "processing_time_ms": processing_time_ms,
                },
            )

            return response

        except Exception as e:
            logger.error(
                f"Orchestrator error: {str(e)}",
                extra={"user_id": request.user_id},
                exc_info=True,
            )

            return RAGResponse(
                status=ResponseStatus.ERROR,
                error=f"Orchestrator error: {str(e)}",
                logic_path=PipelineLogic.LOGIC_2,
            )

    async def get_tool_info(self) -> dict:
        """Get information about currently loaded tools (for debugging)."""
        return {
            "llm_provider": self.settings.llm_provider,
            "llm_model": self.settings.llm_model_name,
            "slm_provider": self.settings.slm_provider,
            "slm_model": self.settings.slm_model_name,
            "embeddings_model": self.settings.embeddings_model_name,
            "vectordb_provider": self.settings.vector_db_provider,
            "cache_provider": self.settings.cache_provider,
        }

    async def answer_with_ephemeral_doc(
        self,
        query: str,
        document_text: str,
        filename: str,
        user_id: str,
        session_id: str,
        request_id: str = None,
    ) -> dict:
        """
        ✅ CORRECT IMPLEMENTATION: Answer query using ephemeral document.

        FLOW:
        1. Get session conversation history (NON-EPHEMERAL - persistent per session)
        2. Get SLM summarizer → summarize document text (EPHEMERAL summary)
        3. Build augmentation context: summary + query + session history
        4. Generate answer via LLM
        5. Update session history with Q&A (NON-EPHEMERAL)
        6. Return answer + sources

        KEY ARCHITECTURE:
        - Document text: EPHEMERAL (not stored in vector DB)
        - Document summary: EPHEMERAL (generated, discarded after query)
        - Session history: NON-EPHEMERAL (persisted in database)
        - NO chunking, NO embedding, NO vector DB storage
        - Only summary is used for augmentation

        Args:
            query: User question about the document (5-5000 chars)
            document_text: Extracted text from document
            filename: Original filename (pdf, txt, md, docx)
            user_id: User identifier
            session_id: Session/chat ID for conversation context
            request_id: Optional request ID for logging correlation

        Returns:
            Dictionary:
            {
                "answer": "Generated answer text",
                "sources": [
                    {
                        "filename": "document.pdf",
                        "section": "Introduction",
                        "text": "relevant excerpt"
                    }
                ],
                "session_id": "session456",
                "user_id": "user123",
            }

        Raises:
            ValueError: Invalid document or empty content
            RAGPipelineException: Pipeline error during processing

        Latency:
            Typical: 2-4 seconds
        """
        request_id = request_id or "ephemeral-doc"

        try:
            logger.info(
                f"Processing ephemeral doc [{request_id}]: {filename}",
                extra={
                    "request_id": request_id,
                    "filename": filename,
                    "user_id": user_id,
                    "session_id": session_id,
                    "doc_text_len": len(document_text),
                },
            )

            # ============================================================
            # STEP 1: Get session conversation history (NON-EPHEMERAL)
            # ============================================================

            logger.debug(
                f"Fetching session history [{request_id}]: {session_id}"
            )

            # Get non-ephemeral session history from state manager
            if hasattr(self, "state_manager") and self.state_manager:
                session_history = (
                    await self.state_manager.get_session(session_id) or []
                )
            else:
                # If no state manager, initialize empty (no persistence)
                logger.warning(f"No state_manager available [{request_id}]")
                session_history = []

            logger.debug(
                f"Session history retrieved [{request_id}]: "
                f"{len(session_history)} messages"
            )

            # ============================================================
            # STEP 2: Validate extracted document text
            # ============================================================

            if not document_text or len(document_text.strip()) == 0:
                logger.error(f"Empty document text [{request_id}]: {filename}")
                raise ValueError(
                    f"Document contains no readable text: {filename}"
                )

            logger.debug(
                f"Document text validated [{request_id}]: "
                f"{len(document_text)} chars"
            )

            # ============================================================
            # STEP 3: Get SLM summarizer → summarize document (EPHEMERAL)
            # ============================================================

            logger.debug(f"Getting SLM summarizer [{request_id}]")

            slm_provider = self.container.get_slm()

            if not slm_provider:
                logger.error(f"SLM provider not available [{request_id}]")
                raise RAGPipelineException(
                    error_code="SLM_NOT_AVAILABLE",
                    message="SLM summarizer not available",
                )

            logger.debug(
                f"Generating document summary via SLM [{request_id}]"
            )

            # Summarize full document (no chunking)
            document_summary = await slm_provider.summarize(
                text=document_text,
                max_length=500,  # Keep summary concise
            )

            if not document_summary:
                logger.warning(f"Empty summary returned [{request_id}]")
                document_summary = (
                    document_text[:500]  # Fallback: use first 500 chars
                )

            logger.info(
                f"Document summarized [{request_id}]: {len(document_summary)} chars",
                extra={
                    "request_id": request_id,
                    "summary_len": len(document_summary),
                },
            )

            # ============================================================
            # STEP 4: Build augmentation context
            # ============================================================

            logger.debug(f"Building augmentation context [{request_id}]")

            # Context = summary + query (summary is EPHEMERAL summary only)
            augmentation_context = f"""Document Summary:
{document_summary}

User Query:
{query}"""

            # Include recent session history for conversational context
            conversation_context = ""
            if session_history:
                recent_msgs = session_history[
                    -10:
                ]  # Last 10 messages for context
                conversation_context = "\n\n---\n\nConversation History:\n"
                for msg in recent_msgs:
                    role = msg.get("role", "unknown").upper()
                    content = msg.get("content", "")
                    conversation_context += f"\n{role}: {content}"

            full_context = augmentation_context + conversation_context

            logger.debug(
                f"Augmentation context built [{request_id}]: "
                f"{len(full_context)} chars"
            )

            # ============================================================
            # STEP 5: Generate answer via LLM
            # ============================================================

            logger.debug(f"Getting LLM provider [{request_id}]")

            llm_provider = self.container.get_llm()

            if not llm_provider:
                logger.error(f"LLM provider not available [{request_id}]")
                raise RAGPipelineException(
                    error_code="LLM_NOT_AVAILABLE",
                    message="LLM provider not available",
                )

            # Build messages for generation
            system_prompt = (
                "You are a helpful assistant answering questions about a document. "
                "Use the provided document summary and conversation history to provide "
                "accurate and concise answers. When referencing the document, "
                "mention specific sections if available."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_context},
            ]

            logger.debug(f"Calling LLM for answer generation [{request_id}]")

            answer = await llm_provider.generate(
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )

            if not answer:
                logger.error(f"Empty answer generated [{request_id}]")
                raise RAGPipelineException(
                    error_code="GENERATION_FAILED",
                    message="Failed to generate answer",
                )

            logger.info(
                f"Answer generated [{request_id}]: {len(answer)} chars",
                extra={
                    "request_id": request_id,
                    "answer_len": len(answer),
                },
            )

            # ============================================================
            # STEP 6: Update session history (NON-EPHEMERAL)
            # ============================================================

            logger.debug(
                f"Updating session history [{request_id}]: {session_id}"
            )

            # Add this Q&A to session history (for future context)
            session_history.append(
                {
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "source": "ephemeral_doc",
                }
            )

            session_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "source": "ephemeral_doc",
                }
            )

            # Save session history (NON-EPHEMERAL)
            if hasattr(self, "state_manager") and self.state_manager:
                await self.state_manager.save_session(
                    session_id, session_history
                )
                logger.info(
                    f"Session history updated [{request_id}]: "
                    f"{len(session_history)} messages"
                )

            # ============================================================
            # STEP 7: Return result (document summary is EPHEMERAL)
            # ============================================================

            # Document summary is not persisted - it's ephemeral
            # Only the Q&A exchange is saved in session history
            
            result = {
                "answer": answer,
                "sources": [
                    {
                        "filename": filename,
                        "type": "ephemeral_document",
                        "summary": document_summary[:200],  # Preview only
                    }
                ],
                "session_id": session_id,
                "user_id": user_id,
                "ephemeral": True,
            }

            logger.info(
                f"Ephemeral doc Q&A complete [{request_id}]",
                extra={
                    "request_id": request_id,
                    "session_id": session_id,
                    "answer_len": len(answer),
                    "filename": filename,
                },
            )

            return result

        except RAGPipelineException as e:
            logger.error(
                f"RAG pipeline error processing ephemeral doc [{request_id}]: "
                f"{e.error_code}",
                extra={
                    "request_id": request_id,
                    "filename": filename,
                    "error_code": e.error_code,
                    "error_message": e.message,
                },
            )
            raise

        except Exception as e:
            logger.error(
                f"Error processing ephemeral doc [{request_id}]: {str(e)}",
                extra={
                    "request_id": request_id,
                    "filename": filename,
                    "user_id": user_id,
                    "session_id": session_id,
                },
                exc_info=True,
            )
            raise
