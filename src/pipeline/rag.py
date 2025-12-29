# LINE 3 orchestrator (combined)
"""
================================================================================
FILE: src/pipelines/rag.py
================================================================================

PURPOSE:
    RAG (Retrieval-Augmented Generation) Pipeline.
    Retrieves relevant documents using embeddings, then generates answer using LLM.
    
    Variants:
    - LOGIC_2: Pure RAG (no document, just context)
    - LOGIC_4: RAG + Document (with user-provided document context)

WORKFLOW (LOGIC_2/4 - RAG Pipeline):
    1. Embed user query
    2. Search Vector DB for relevant documents
    3. Rank results by similarity
    4. Pass top results as context to LLM
    5. LLM generates answer
    6. Cache answer in Redis
    7. Return to user

LATENCY:
    - LOGIC_2 (Pure RAG): 1-2s
    - LOGIC_4 (RAG + Document): 1-2s (document context already prepared)

IMPORTS:
    - asyncio: Async execution
    - logging: Logging
    - base.BasePipeline: Base class
    - core handlers: embeddings, vector_db, llm, redis
    - schemas: Request/response models

INPUTS:
    - request: UserQueryRequest (prompt, optional document summary)
    - handlers: embeddings_handler, vector_db_handler, llm_handler, redis_handler
    - models: llm model for inference

OUTPUTS:
    - RAGResponse with answer + sources
    - Answer cached in Redis

RAG FLOW:
    User Query
        ↓
    Embed Query (embeddings_handler)
        ↓
    Search Vector DB (vector_db_handler)
        ↓
    Get Top-K Results + Filter by Score
        ↓
    Format as Context
        ↓
    Generate Answer (llm_handler)
        ↓
    Cache Answer (redis_handler)
        ↓
    Return Response

KEY FACTS:
    - Semantic search (embedding-based)
    - Context-aware LLM generation
    - Source attribution (show where answer came from)
    - Results ranked by relevance

FUTURE SCOPE (Phase 2+):
    - Add hybrid search (keyword + semantic)
    - Add re-ranking (reorder results)
    - Add multi-stage retrieval
    - Add feedback loop (user rating answers)
    - Add A/B testing (different retrievers)

TESTING ENVIRONMENT:
    - Mock embeddings model
    - Mock vector DB responses
    - Mock LLM responses
    - Test end-to-end flow

PRODUCTION DEPLOYMENT:
    - Monitor latency (target: <2s)
    - Monitor hit rate (source relevance)
    - Monitor LLM quality (user ratings)
    - Cache all answers
"""

# ================================================================================
# IMPORTS
# ================================================================================

import logging
import time
from typing import List, Dict, Any

from src.pipeline.base import BasePipeline
from .schemas import (
    UserQueryRequest, RAGResponse, ResponseStatus, PipelineLogic,
    RAGResult, SourceAttribution
)
from src.core.exceptions import RAGPipelineException
from src.config.settings import Settings
from src.container.service_container import ServiceContainer

from src.core import (
    RedisHandler,
    VectorDBHandler,
    EmbeddingsHandler,
    LLMHandler,
    SLMHandler,
)

logger = logging.getLogger(__name__)

# ================================================================================
# RAG PIPELINE CLASS
# ================================================================================

class RAGPipeline(BasePipeline):
    """
    RAG (Retrieval-Augmented Generation) Pipeline.
    
    Retrieves relevant documents and generates answer using LLM.
    """
    class RAGPipeline:
        def __init__(self, settings: Settings, container: ServiceContainer):
            """RAG pipeline orchestrating LLM, SLM, embeddings, vector DB, and Redis."""
            self.settings = settings
            self.container = container
            # DO NOT require individual handlers here
            # handlers dict will be built lazily

    def _ensure_handlers(self) -> None:
        """Initialize handler dictionary from the service container once."""
        if hasattr(self, "handlers"):
            return

        self.handlers = {
            "llm": self.container._llm,
            "slm": self.container._slm,
            "embeddings": self.container._embeddings,
            "vector_db": self.container._vectordb,
            "redis": self.container._cache,
        }


    async def execute(self, request: UserQueryRequest, logic_path: PipelineLogic) -> RAGResponse:
        """
        Execute RAG pipeline.
        
        Args:
            request: User query request
            logic_path: LOGIC_2 or LOGIC_4
        
        Returns:
            RAGResponse with answer + sources
        
        LATENCY: 1-2s
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not await self.validate_input(request):
                return await self.handle_error(request, ValueError("Invalid input"), logic_path)
            
            logger.info(f"Starting RAG pipeline ({logic_path})")
            
            # STEP 1: Embed query
            self._ensure_handlers()
            embeddings_handler = self.handlers["embeddings"]
            query_embedding = await embeddings_handler.embed_query(request.prompt)
            
            logger.debug(f"Query embedded: {len(query_embedding)} dimensions")
            
            # STEP 2: Search vector DB
            vector_db_handler = self.handlers["vector_db"]
            search_results = await vector_db_handler.search(
                collection_name="documents",
                query_vector=query_embedding,
                limit=self.settings.vector_db_top_k,
                score_threshold=self.settings.vector_db_score_threshold
            )
            
            logger.debug(f"Retrieved {len(search_results)} documents from vector DB")
            
            # STEP 3: Format context from search results
            context = self._format_context(search_results)
            source_attributions = self._extract_sources(search_results)
            
            # STEP 4: Generate answer using LLM
            llm_handler = self.handlers["llm"]
            answer = await llm_handler.generate(
                prompt=request.prompt,
                context=context,
                max_tokens=500,
                temperature=0.7
            )
            
            logger.debug(f"Answer generated: {len(answer)} chars")
            
            # STEP 5: Cache answer
            redis_handler = self.handlers["redis"]
            await redis_handler.set_query_cache(
                user_id=request.user_id,
                query=request.prompt,
                answer=answer,
                ttl=3600  # 1 hour
            )
            
            # STEP 6: Build response
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = RAGResult(
                answer=answer,
                sources=source_attributions,
                processing_time_ms=processing_time_ms
            )
            
            response = RAGResponse(
                status=ResponseStatus.SUCCESS,
                result=result,
                logic_path=logic_path
            )
            
            await self.log_execution(request, response, processing_time_ms)
            return response
        
        except Exception as e:
            return await self.handle_error(request, e, logic_path)
    
    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results as context for LLM"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            payload = result.get("payload", {})
            text = payload.get("text", "")
            score = result.get("score", 0)
            
            context_parts.append(f"Document {i} (relevance: {score:.2f}):\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[SourceAttribution]:
        """Extract source attributions from search results"""
        sources = []
        
        for result in search_results:
            source = SourceAttribution(
                source_id=str(result.get("id", "unknown")),
                relevance_score=result.get("score", 0.0),
                excerpt=result.get("payload", {}).get("text", "")[:200]
            )
            sources.append(source)
        
        return sources

# ================================================================================
# FUTURE EXTENSION (Phase 2+)
# ================================================================================

# TODO (Phase 2): Add hybrid search
# TODO (Phase 2): Add re-ranking
# TODO (Phase 2): Add multi-stage retrieval
# TODO (Phase 2): Add feedback loop
