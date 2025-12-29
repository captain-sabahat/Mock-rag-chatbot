# Route to LOGIC_1/2/3/4
"""
================================================================================
FILE: src/pipelines/logic_router.py
================================================================================

PURPOSE:
    Decision logic to determine which pipeline (LOGIC_1/2/3/4) to execute.
    Routes based on: redis_lookup flag, document_attached flag, cache hit.

WORKFLOW:
    1. Receive UserQueryRequest
    2. Check redis_lookup flag (should we try cache?)
    3. Check document_attached flag (is doc in request?)
    4. Determine logic path:
       - LOGIC_1: redis_lookup=YES, HIT → ultra-fast cache path
       - LOGIC_2: redis_lookup=NO, no doc → pure RAG path
       - LOGIC_3: redis_lookup=YES, HIT with doc → cache + doc
       - LOGIC_4: redis_lookup=NO, doc → RAG + doc
    5. Return logic path + decision reason

LOGIC DECISION TREE:
    Does user want cache? (redis_lookup flag)
        └─ YES
            └─ Does cache have answer? (check Redis)
                ├─ YES → LOGIC_1 (ultra-fast, <10ms)
                └─ NO → Continue
        └─ NO → Continue
    
    Is document attached? (doc_attached flag)
        ├─ YES
        │   └─ Run full RAG + summarization → LOGIC_3 or LOGIC_4
        │       ├─ If cache hit → LOGIC_3
        │       └─ If cache miss → LOGIC_4
        └─ NO
            └─ Run pure RAG (no doc) → LOGIC_2

IMPORTS:
    - asyncio: Async operations
    - logging: Logging
    - schemas: Request/response models
    - core handlers: Redis, state manager

INPUTS:
    - request: UserQueryRequest
    - redis_handler: For cache lookup
    - state_manager: For session state

OUTPUTS:
    - logic_path: Which pipeline to execute (LOGIC_1/2/3/4)
    - cache_result: Result if cache hit (LOGIC_1)
    - reason: Why this logic path was chosen (for debugging)

LATENCY:
    - Decision logic: <5ms (just lookup, no I/O)

KEY FACTS:
    - Fast decision logic (no heavy computation)
    - Determines pipeline without executing it
    - Enables optimal latency per request
    - Simplifies main orchestrator

FUTURE SCOPE (Phase 2+):
    - Add A/B testing (route to different models)
    - Add user preference routing
    - Add ML-based routing (predict best path)
    - Add load-based routing (scale to fewer backends)
    - Add cost-aware routing (cheaper vs faster)

TESTING ENVIRONMENT:
    - Test all logic paths
    - Verify cache lookup
    - Test document detection
    - Verify reasons logged

PRODUCTION DEPLOYMENT:
    - Fast path for LOGIC_1 (cache hit)
    - Optimal routing for all users
    - Metrics on which logic paths used
    - Monitor cache hit rate

WORKFLOW:

1. Receive UserQueryRequest
2. Check redis_lookup flag (should we try cache?)
3. Check document_attached flag (is doc in request?)
4. Determine logic path:

   - LOGIC_1: redis_lookup=YES, HIT  → ultra-fast cache path
   - LOGIC_2: redis_lookup=NO, no doc → pure RAG path
   - LOGIC_3: redis_lookup=YES, HIT with doc → cache + doc
   - LOGIC_4: redis_lookup=NO, doc → RAG + doc

5. Return logic path + decision reason
"""

# ================================================================================
# IMPORTS
# ================================================================================

import logging
from typing import Tuple, Optional

from .schemas import UserQueryRequest, PipelineLogic
from src.config.constants import CACHE_KEY_QUERY
from src.utils import hash_query

logger = logging.getLogger(__name__)

# ================================================================================
# LOGIC ROUTER CLASS
# ================================================================================


class LogicRouter:
    """
    Routes requests to appropriate pipeline based on flags and cache state.
    """

    def __init__(self, settings, cache=None) -> None:
        """
        Initialize logic router.

        Args:
            settings: Settings instance (kept for future use / logging).
            cache: Cache provider implementing ICacheProvider interface.
                   This can be RedisCacheProvider or NoOpCacheProvider.
        """
        self.settings = settings
        self.cache = cache  # expected to expose async get(key)

    async def route(
        self,
        request: UserQueryRequest,
    ) -> Tuple[PipelineLogic, Optional[str], str]:
        """
        Determine which logic path to use.

        Args:
            request: User query request

        Returns:
            Tuple of:
              - logic_path: Which pipeline (LOGIC_1/2/3/4)
              - cache_result: Result if cache hit (None otherwise)
              - reason: Why this path was chosen (for logging)
        """
        # Check if user wants to use cache
        cache_enabled = request.redis_lookup.value == "yes"
        doc_attached = request.doc_attached.value == "yes"

        cache_result: Optional[str] = None

        # Try cache if enabled and cache provider is available
        if cache_enabled and self.cache is not None:
            try:
                cache_result = await self._check_cache(
                    user_id=request.user_id,
                    query=request.prompt,
                )
                if cache_result:
                    logger.debug(f"Cache HIT for user {request.user_id}")

                    if doc_attached:
                        return (
                            PipelineLogic.LOGIC_3,
                            cache_result,
                            "Cache HIT + Document attached",
                        )
                    else:
                        return (
                            PipelineLogic.LOGIC_1,
                            cache_result,
                            "Cache HIT + Pure query",
                        )
                else:
                    logger.debug(f"Cache MISS for user {request.user_id}")
            except Exception as e:
                logger.warning(
                    f"Cache check failed: {str(e)}, falling back to RAG",
                    exc_info=True,
                )
                cache_result = None

        # No cache hit or cache disabled → route based on document flag
        if doc_attached:
            return (
                PipelineLogic.LOGIC_4,
                None,
                "Cache MISS/disabled + Document attached",
            )
        else:
            return (
                PipelineLogic.LOGIC_2,
                None,
                "Cache MISS/disabled + Pure query",
            )

    async def _check_cache(self, user_id: str, query: str) -> Optional[str]:
        """
        Check cache for query result.

        Args:
            user_id: User ID
            query: Query string

        Returns:
            Cached result or None if not found
        """
        if self.cache is None:
            return None

        query_hash = hash_query(query)
        key = CACHE_KEY_QUERY.format(user_id=user_id, query_hash=query_hash)

        try:
            result = await self.cache.get(key)
            return result
        except Exception as e:
            logger.error(f"Cache check failed for key '{key}': {str(e)}")
            return None


# ================================================================================
# FUTURE EXTENSION (Phase 2+)
# ================================================================================

# TODO (Phase 2): Add A/B testing routing
# TODO (Phase 2): Add user preference routing
# TODO (Phase 2): Add ML-based routing
# TODO (Phase 2): Add load-based routing
# TODO (Phase 2): Add cost-aware routing
