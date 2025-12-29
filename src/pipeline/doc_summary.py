# LINE 1 orchestrator
"""
================================================================================
FILE: src/pipelines/doc_summary.py
================================================================================

PURPOSE:
    LOGIC_1 Pipeline: Document summarization line.
    Triggered when user attaches document.
    Creates concise summary for RAG context.

WORKFLOW (LINE 1 - Document Summarization):
    1. Receive document (PDF/DOCX/TXT)
    2. Parse document → text extraction
    3. Chunk document (1000 chars per chunk)
    4. Summarize using SLM (2-5s)
    5. Store summary in Redis cache (7 days TTL)
    6. Return summary to orchestrator

LATENCY: 2-5s (dominated by SLM inference)

IMPORTS:
    - asyncio: Async execution
    - logging: Logging
    - base.BasePipeline: Base class
    - core handlers: doc_ingestion, slm_handler, redis_handler
    - schemas: Request/response models

INPUTS:
    - request.document: DocumentAttachment (file_content, file_type, file_name)
    - handlers: doc_ingestion, slm_handler, redis_handler

OUTPUTS:
    - RAGResponse with summary
    - Summary cached in Redis

KEY FACTS:
    - Only triggered when document attached
    - Heavy on SLM (2-5s inference)
    - Result cached (7 days TTL)
    - Used by downstream RAG pipeline

FUTURE SCOPE (Phase 2+):
    - Add multi-document summarization
    - Add incremental updates (doc changed → update summary)
    - Add extractive + abstractive hybrid
    - Add topic detection
    - Add key phrase extraction

TESTING ENVIRONMENT:
    - Test with sample PDF/DOCX/TXT
    - Mock SLM for faster tests
    - Verify summary caching
    - Test error handling

PRODUCTION DEPLOYMENT:
    - Monitor SLM latency
    - Alert if >5s
    - Cache all summaries
    - Monitor cache hit rate
"""

# ================================================================================
# IMPORTS
# ================================================================================

import logging
import time
from src.pipeline.base import BasePipeline
from src.pipeline.schemas import UserQueryRequest, RAGResponse, ResponseStatus, PipelineLogic
from src.core.exceptions import DocumentProcessingError, SLMError

logger = logging.getLogger(__name__)

# ================================================================================
# DOC SUMMARY PIPELINE CLASS
# ================================================================================

class DocSummaryPipeline(BasePipeline):
    """
    Document summarization pipeline (LINE 1).
    
    Parses document and creates summary for RAG context.
    """
    
    async def execute(self, request: UserQueryRequest) -> RAGResponse:
        """
        Execute document summarization pipeline.
        
        Args:
            request: User query request (with document attachment)
        
        Returns:
            RAGResponse with summary
        
        LATENCY: 2-5s
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not await self.validate_input(request):
                return await self.handle_error(
                    request,
                    ValueError("Invalid input"),
                    PipelineLogic.LOGIC_1
                )
            
            # Check document attachment
            if not request.document:
                return await self.handle_error(
                    request,
                    ValueError("No document attached"),
                    PipelineLogic.LOGIC_1
                )
            
            logger.info(f"Summarizing document: {request.document.file_name}")
            
            # STEP 1: Parse document
            doc_ingestion = self.handlers["doc_ingestion"]
            text = await doc_ingestion.parse_document(
                file_content=request.document.file_content.encode(),
                file_type=request.document.file_type,
                file_name=request.document.file_name
            )
            
            logger.debug(f"Document parsed: {len(text)} chars")
            
            # STEP 2: Summarize using SLM
            slm_handler = self.handlers["slm"]
            summary = await slm_handler.summarize(
                text=text,
                max_length=150,
                min_length=50
            )
            
            logger.debug(f"Document summarized: {len(summary)} chars")
            
            # STEP 3: Cache summary
            redis_handler = self.handlers["redis"]
            from src.utils import hash_document
            doc_hash = hash_document(request.document.file_content.encode())
            from src.config.cache_config import CACHE_CONFIG
            cache_key = CACHE_CONFIG.keys.summary_key(request.user_id, doc_hash)
            
            await redis_handler.set(
                cache_key,
                summary,
                ttl=604800  # 7 days
            )
            
            # STEP 4: Return response
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            response = RAGResponse(
                status=ResponseStatus.SUCCESS,
                result={
                    "summary": summary,
                    "processing_time_ms": processing_time_ms
                },
                logic_path=PipelineLogic.LOGIC_1
            )
            
            await self.log_execution(request, response, processing_time_ms)
            return response
        
        except Exception as e:
            return await self.handle_error(request, e, PipelineLogic.LOGIC_1)

# ================================================================================
# FUTURE EXTENSION (Phase 2+)
# ================================================================================

# TODO (Phase 2): Add multi-document summarization
# TODO (Phase 2): Add incremental updates
# TODO (Phase 2): Add extractive + abstractive hybrid
# TODO (Phase 2): Add topic detection
