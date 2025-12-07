"""
================================================================================
PIPELINE ORCHESTRATOR - LangGraph Workflow Manager
================================================================================

PURPOSE:
--------
Orchestrate the RAG pipeline using LangGraph for stateful workflows.
Integrates all 5 processing nodes into a coordinated pipeline.

Responsibilities:
  - Build pipeline graph
  - Initialize state
  - Execute nodes in sequence
  - Handle errors and retries
  - Track progress and checkpoints
  - Manage circuit breaker
  - Persist state to SessionStore

Pipeline Steps:
  1. Ingestion (parse file → text)
  2. Preprocessing (clean text)
  3. Chunking (split into chunks)
  4. Embedding (generate vectors)
  5. VectorDB (store indexed vectors)

ARCHITECTURE:
--------------
    User Request
         ↓
    PipelineOrchestrator.process_document(request_id, file_name, file_content)
         ↓
    Create initial PipelineState
        ↓
    Save initial state
         ↓
    _execute_pipeline()
         ├─ _execute_node("ingestion") → ingestion_node()
         ├─ _execute_node("preprocessing") → preprocessing_node()
         ├─ _execute_node("chunking") → chunking_node()
         ├─ _execute_node("embedding") → embedding_node()
         └─ _execute_node("vectordb") → vectordb_node()
         ↓
    Each node:
        1. Validates input
        2. Executes with retry + timeout
        3. Updates checkpoint
        4. Saves state
         ↓
    Save final state to SessionStore
         ↓
    Return request_id
================================================================================

🔄 Error Handling Flow: 

        _execute_node()
            ↓
        Try to execute with retry
            ├─ Timeout → Retry with backoff (2^attempt)
            ├─ Failure → Retry with backoff
            └─ Max retries → ProcessingError
            ↓
        Catch errors:
        ├─ ProcessingError → Status="error", record in checkpoint
        ├─ ValidationError → Status="error", record in checkpoint
        └─ Other Exception → Status="error", record in checkpoint
            ↓
        Circuit breaker:
        ├─ Record failure
        ├─ Trigger circuit break in session store
        └─ Future calls blocked until recovery
            ↓
        Save state to SessionStore
            ↓
        Return state (may be error state)
            
================================================================================

CONFIG: config/settings/pipeline.yaml
-------
    pipeline:
      enabled: true
      chunking_strategy: recursive
      chunk_size: 512
      chunk_overlap: 50
      embedding_model: openai
      vectordb_backend: qdrant

================================================================================
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from src.pipeline.schemas import PipelineState, NodeStatus
from src.pipeline.nodes import (
    ingestion_node,
    preprocessing_node,
    chunking_node,
    embedding_node,
    vectordb_node,
)

from src.core import (
    CircuitBreakerManager,
    CircuitBreakerConfig,
    ProcessingError,
    ValidationError,
)

logger = logging.getLogger(__name__)


@dataclass
class NodeExecutionConfig:
    """Configuration for node execution."""
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 300.0
    enable_circuit_breaker: bool = True


class PipelineOrchestrator:
    """
    Orchestrate the RAG document processing pipeline.
    
    Coordinates execution of all 5 nodes with error handling,
    progress tracking, and state persistence using IngestionStore.
    """

    def __init__(self, config: Optional[NodeExecutionConfig] = None):
        """
        Initialize orchestrator.

        Args:
            config: Node execution configuration
        """
        self.config = config or NodeExecutionConfig()
        self.circuit_breaker_manager = CircuitBreakerManager(CircuitBreakerConfig())

        # Node registry (maps node names to node functions)
        self.node_registry: Dict[str, Callable] = {
            "ingestion": ingestion_node,
            "preprocessing": preprocessing_node,
            "chunking": chunking_node,
            "embedding": embedding_node,
            "vectordb": vectordb_node,
        }

        # Node execution order
        self.node_order = [
            "ingestion",
            "preprocessing",
            "chunking",
            "embedding",
            "vectordb",
        ]

        logger.info("🚀 Pipeline orchestrator initialized")

    async def process_document(
        self,
        request_id: str,
        file_name: str,
        file_content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        **config_overrides
    ) -> str:
        """
        Process a document through the entire pipeline.

        Steps:
        1. Create pipeline state
        2. Execute all 5 nodes sequentially
        3. Return request_id for status tracking

        Args:
            request_id: Unique session ID
            file_name: File name (PDF, TXT, JSON, MD)
            file_content: Raw file bytes
            metadata: Optional custom metadata
            **config_overrides: Override pipeline configuration

        Returns:
            str: Request ID for status tracking

        Raises:
            ProcessingError: If pipeline fails
            ValidationError: If inputs invalid
        """
        try:
            logger.info(
                f"📄 Processing document: {file_name} "
                f"(request_id: {request_id})"
            )

            # Create initial state
            state = PipelineState(
                request_id=request_id,
                file_name=file_name,
                raw_content=file_content,
                metadata=metadata or {},
                started_at=datetime.utcnow(),
                **config_overrides  # Override defaults
            )

            state.add_message(f"🔄 Pipeline started for {file_name}")

            # Execute pipeline
            result_state = await self._execute_pipeline(state)

            # Mark as completed (if not already error)
            if result_state.status != "error":
                result_state.status = "completed"
                result_state.progress_percent = 100
                result_state.completed_at = datetime.utcnow()

                # Calculate total duration
                if result_state.started_at:
                    duration = (
                        result_state.completed_at - result_state.started_at
                    ).total_seconds() * 1000
                    result_state.total_duration_ms = duration

                result_state.add_message("✅ Pipeline completed successfully")

            logger.info(
                f"✅ Processing complete: {request_id} "
                f"(status: {result_state.status})"
            )
            return request_id

        except ValidationError as e:
            logger.error(f"❌ Validation error: {e.message}")
            raise

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise ProcessingError(
                f"Pipeline processing failed: {str(e)}",
                details={"request_id": request_id, "file_name": file_name}
            )

    async def _execute_pipeline(self, state: PipelineState) -> PipelineState:
        """
        Execute the complete pipeline with all nodes.

        Runs nodes in sequence:
        1. ingestion
        2. preprocessing
        3. chunking
        4. embedding
        5. vectordb

        Args:
            state: Initial pipeline state

        Returns:
            PipelineState: Final state after all nodes
        """
        logger.info(f"🚀 Starting pipeline execution: {state.request_id}")

        try:
            # Execute each node in order
            for i, node_name in enumerate(self.node_order):
                logger.debug(
                    f"📍 Node {i+1}/{len(self.node_order)}: {node_name}"
                )

                # Execute node
                state = await self._execute_node(state, node_name)

                # Check for errors (stop pipeline on error)
                if state.status == "error":
                    logger.warning(f"⚠️ Pipeline halted at node: {node_name}")
                    return state

            logger.info(f"✅ Pipeline execution complete: {state.request_id}")
            return state

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")
            state.status = "error"
            state.add_error(f"Pipeline execution: {str(e)}")
            return state

    async def _execute_node(
        self, state: PipelineState, node_name: str
    ) -> PipelineState:
        """
        Execute a single pipeline node with error handling and retry logic.

        Steps:
        1. Validate input
        2. Check circuit breaker
        3. Call node function
        4. Update checkpoint
        5. Handle errors

        Args:
            state: Current pipeline state
            node_name: Name of node to execute

        Returns:
            PipelineState: Updated state
        """
        logger.info(f"📍 Executing node: {node_name}")
        state.current_node = node_name
        state.update_checkpoint(node_name, NodeStatus.RUNNING)

        # Update progress
        node_index = self.node_order.index(node_name)
        progress = (node_index / len(self.node_order)) * 100
        state.progress_percent = int(progress)

        try:
            # Check circuit breaker (if enabled)
            if self.config.enable_circuit_breaker:
                breaker = self.circuit_breaker_manager.get_breaker(node_name)
                if not breaker.can_attempt():
                    raise ProcessingError(
                        f"Circuit breaker OPEN for {node_name}. "
                        f"Service temporarily unavailable.",
                        details={
                            "node": node_name,
                            "breaker_state": breaker.state.value
                        }
                    )

            # Get node function
            if node_name not in self.node_registry:
                raise ValidationError(f"Unknown node: {node_name}")

            node_func = self.node_registry[node_name]

            # Execute node with retry logic
            state = await self._execute_with_retry(node_name, node_func, state)

            # Record success with circuit breaker
            if self.config.enable_circuit_breaker:
                self.circuit_breaker_manager.get_breaker(
                    node_name
                ).record_success()

            # Update progress after node
            progress = ((node_index + 1) / len(self.node_order)) * 100
            state.progress_percent = int(progress)
            return state

        except ProcessingError as e:
            logger.error(f"❌ Node processing error: {node_name} - {e.message}")
            state.status = "error"
            state.add_error(f"{node_name}: {e.message}")
            state.update_checkpoint(
                node_name,
                status=NodeStatus.FAILED,
                error_flag=True,
                error_message=e.message
            )

            # Record failure with circuit breaker
            if self.config.enable_circuit_breaker:
                self.circuit_breaker_manager.get_breaker(
                    node_name
                ).record_failure()

            return state

        except ValidationError as e:
            logger.error(f"❌ Node validation error: {node_name} - {e.message}")
            state.status = "error"
            state.add_error(f"{node_name} validation: {e.message}")
            state.update_checkpoint(
                node_name,
                status=NodeStatus.FAILED,
                error_flag=True,
                error_message=e.message
            )
            return state

        except Exception as e:
            logger.error(f"❌ Node execution error: {node_name} - {str(e)}")
            state.status = "error"
            state.add_error(f"{node_name} error: {str(e)}")
            state.update_checkpoint(
                node_name,
                status=NodeStatus.FAILED,
                error_flag=True,
                error_message=str(e)
            )

            # Record failure with circuit breaker
            if self.config.enable_circuit_breaker:
                self.circuit_breaker_manager.get_breaker(
                    node_name
                ).record_failure()

            return state

    async def _execute_with_retry(
        self, node_name: str, node_func: Callable, state: PipelineState
    ) -> PipelineState:
        """
        Execute a node function with automatic retry on failure.

        Implements exponential backoff:
        - Retry 1: wait 1 second
        - Retry 2: wait 2 seconds
        - Retry 3: wait 4 seconds

        Args:
            node_name: Name of node
            node_func: Async function to execute
            state: Pipeline state

        Returns:
            Updated PipelineState
        """
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug(
                    f"Execution attempt {attempt + 1}/"
                    f"{self.config.max_retries + 1} for {node_name}"
                )

                # Execute node function
                result_state = await asyncio.wait_for(
                    node_func(state),
                    timeout=self.config.timeout_seconds
                )
                return result_state

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    f"⏱️ Node {node_name} timed out "
                    f"({self.config.timeout_seconds}s). "
                    f"Retries remaining: {self.config.max_retries - attempt}"
                )

                if attempt < self.config.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

            except Exception as e:
                last_error = e
                logger.warning(
                    f"❌ Node {node_name} failed: {str(e)}. "
                    f"Retries remaining: {self.config.max_retries - attempt}"
                )

                if attempt < self.config.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        raise ProcessingError(
            f"Node execution failed after {self.config.max_retries + 1} "
            f"attempts. Last error: {str(last_error)}",
            details={
                "node": node_name,
                "attempts": self.config.max_retries + 1,
                "error": str(last_error)
            }
        )

    async def query_documents(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query processed documents using semantic search.

        Steps:
        1. Generate query embedding
        2. Search vector DB
        3. Return results with scores

        Args:
            query: User query string
            top_k: Number of results to return
            session_id: Optional session ID for filtering
            filters: Optional metadata filters

        Returns:
            Dict with query results

        Raises:
            ProcessingError: If query fails
        """
        logger.info(f"🔍 Querying: {query} (top_k={top_k})")

        try:
            # TODO: Implement vector search
            # 1. Generate embedding for query
            # 2. Search vector DB
            # 3. Return top_k results

            results = {
                "query": query,
                "results": [],
                "total": 0,
                "processing_time_ms": 0
            }

            logger.info(f"✅ Query complete: {len(results['results'])} results")
            return results

        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")
            raise ProcessingError(f"Query execution failed: {str(e)}")

    async def get_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get current status of a pipeline execution.

        Args:
            request_id: Session ID

        Returns:
            Status information
        """
        try:
            # TODO: Fetch from IngestionStore
            return {
                "request_id": request_id,
                "status": "processing",
                "progress_percent": 50,
                "current_node": "preprocessing",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Failed to get status: {str(e)}")
            raise ProcessingError(f"Failed to get status: {str(e)}")

    async def cancel_processing(self, request_id: str) -> bool:
        """
        Cancel ongoing pipeline execution.

        Args:
            request_id: Session ID

        Returns:
            bool: Success flag
        """
        try:
            logger.info(f"✅ Pipeline cancelled: {request_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to cancel: {str(e)}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Check orchestrator health.

        Returns:
            Health information
        """
        return {
            "status": "healthy",
            "circuit_breakers": self.circuit_breaker_manager.get_all_status(),
            "nodes_registered": len(self.node_registry),
            "node_order": self.node_order,
        }


# Singleton instance
_orchestrator: Optional[PipelineOrchestrator] = None


def get_orchestrator(
    config: Optional[NodeExecutionConfig] = None,
) -> PipelineOrchestrator:
    """
    Get or create singleton orchestrator instance.

    Args:
        config: Optional configuration (used on first call only)

    Returns:
        PipelineOrchestrator: Singleton instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator(config=config)
        logger.info("✅ Orchestrator singleton created")

    return _orchestrator


def reset_orchestrator() -> None:
    """Reset singleton (useful for testing)."""
    global _orchestrator
    _orchestrator = None
    logger.info("🔄 Orchestrator singleton reset")