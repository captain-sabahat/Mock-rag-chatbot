# LINE 2 orchestrator
"""
================================================================================
FILE: src/pipelines/redis_cache.py
================================================================================

PURPOSE:
    LOGIC_1 Pipeline: Redis cache-only path.
    Ultra-fast path (<10ms) that returns cached query results.
    
    Triggered when:
    - User enables cache lookup (redis_lookup=YES)
    - Previous answer exists in Redis (cache HIT)

WORKFLOW (LOGIC_1 - Cache Path):
    1. Hash user query
    2. Check Redis cache key
    3. If found (HIT): return immediately (<10ms)
    4. If not found (MISS): return error (should route to other logic)

LATENCY: <10ms for HIT

IMPORTS:
    - asyncio: Async execution
    - logging: Logging
    - base.BasePipeline: Base class
    - core handlers: redis_handler
    - schemas: Request/response models

INPUTS:
    - request: UserQueryRequest
    - cached_result: Cached answer (from logic_router)

OUTPUTS:
    - RAGResponse with cached answer

KEY FACTS:
    - Fastest possible path
    - No computation, just lookup
    - Cache-first strategy for repeated queries
    - Reduces load on backends

CACHE HIT RATE:
    - Target: >70% for same user
    - TTL: 1 hour (query cache)
    - Enables ultra-fast responses

FUTURE SCOPE (Phase 2+):
    - Add cache warming (pre-populate common queries)
    - Add cache invalidation patterns
    - Add cache analytics
    - Add personalization per user

TESTING ENVIRONMENT:
    - Mock Redis cache
    - Test hit/miss scenarios
    - Verify fast return

PRODUCTION DEPLOYMENT:
    - Monitor cache hit rate
    - Alert if hit rate drops
    - Optimize TTL based on usage
"""

# ================================================================================
# IMPORTS
# ================================================================================

import logging
import time

from src.pipeline.base import BasePipeline
from .schemas import UserQueryRequest, RAGResponse, ResponseStatus, PipelineLogic, RAGResult
from src.core.exceptions import RAGPipelineException

logger = logging.getLogger(__name__)

# ================================================================================
# REDIS CACHE PIPELINE CLASS
# ================================================================================

class RedisCachePipeline(BasePipeline):
    """
    Redis Cache Pipeline (LOGIC_1).
    
    Ultra-fast path returning cached results.
    """
    
    async def execute(
        self,
        request: UserQueryRequest,
        cached_result: str
    ) -> RAGResponse:
        """
        Execute cache pipeline.
        
        Args:
            request: User query request
            cached_result: Cached answer (from logic_router)
        
        Returns:
            RAGResponse with cached answer
        
        LATENCY: <10ms
        """
        start_time = time.time()
        
        try:
            if not cached_result:
                return await self.handle_error(
                    request,
                    ValueError("No cached result"),
                    PipelineLogic.LOGIC_1
                )
            
            logger.info(f"Returning cached result for {request.user_id}")
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = RAGResult(
                answer=cached_result,
                sources=[],  # No sources for cached result
                processing_time_ms=processing_time_ms
            )
            
            response = RAGResponse(
                status=ResponseStatus.SUCCESS,
                result=result,
                logic_path=PipelineLogic.LOGIC_1
            )
            
            await self.log_execution(request, response, processing_time_ms)
            return response
        
        except Exception as e:
            return await self.handle_error(request, e, PipelineLogic.LOGIC_1)
