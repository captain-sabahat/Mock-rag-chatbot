# Session state + TTL
"""
================================================================================
FILE: src/core/state_manager.py
================================================================================

PURPOSE:
    Manages per-user/per-session state (not cached). Tracks user context,
    conversation history, temporary flags. In-memory storage with TTL.

WORKFLOW:
    1. User creates session
    2. StateManager tracks session state (user_id, session_id)
    3. User asks queries
    4. State persists across requests (same session)
    5. On session end or timeout → state auto-expires (TTL)

STATE CATEGORIES:
    - User context: user_id, preferences
    - Session info: session_id, created_time
    - Conversation history: previous queries/answers
    - Temporary flags: doc_processed, cache_status

STORAGE:
    - In-memory dict (not persistent)
    - TTL-based expiration (auto-cleanup)
    - Per-user/per-session isolation

LATENCY:
    - GET state: <1ms (in-memory)
    - SET state: <1ms (in-memory)
    - Cleanup: Background task

IMPORTS:
    - asyncio: Async operations, timers
    - time: Timestamps
    - logging: Logging

INPUTS:
    - user_id: User identifier
    - session_id: Session identifier
    - state_data: Dict of state key-value pairs

OUTPUTS:
    - State value (dict)
    - Existence flag (True/False)

KEY FACTS:
    - In-memory only (not persistent)
    - Different from Redis (which is persistent cache)
    - Short-lived (session duration)
    - Non-critical data (loss is acceptable)

DIFFERENCE: StateManager vs Redis:
    - StateManager: Session state, temporary, in-memory
    - Redis: Query cache, persistent, disk-backed
    - StateManager: <1ms latency
    - Redis: 1-5ms latency (network I/O)

RESILIENCE:
    - Auto-cleanup on TTL
    - Graceful degradation if state lost
    - No circuit breaker needed (in-memory)

FUTURE SCOPE (Phase 2+):
    - Add persistence (to Redis)
    - Add distributed state (multi-region)
    - Add state snapshots (for debugging)
    - Add state metrics (count, size)
    - Add user preferences persistence

TESTING ENVIRONMENT:
    - Create/destroy state in tests
    - Verify TTL cleanup
    - Test concurrent access

PRODUCTION DEPLOYMENT:
    - Monitor state count (memory usage)
    - Alert if TTL cleanup failing
    - Persist critical state to Redis (Phase 2)
"""

# ================================================================================
# IMPORTS
# ================================================================================

import asyncio
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ================================================================================
# STATE MANAGER CLASS
# ================================================================================

class StateManager:
    """
    Manages per-user/per-session state (in-memory, temporary).
    
    Tracks user context, conversation history, temporary flags.
    Auto-expires state after TTL (default: 24 hours).
    """
    
    def __init__(self, default_ttl_seconds: int = 86400):
        """
        Initialize state manager.
        
        Args:
            default_ttl_seconds: Default TTL for state (24 hours)
        """
        self._state_dict: Dict[str, Dict[str, Any]] = {}
        self._ttl_dict: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._default_ttl = default_ttl_seconds
        
        logger.info(f"StateManager initialized (default TTL: {default_ttl_seconds}s)")
    
    async def get_state(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session state.
        
        Args:
            user_id: User ID
            session_id: Session ID
        
        Returns:
            State dict or None if not found/expired
        
        LATENCY: <1ms
        """
        key = f"{user_id}:{session_id}"
        
        async with self._lock:
            # Check if expired
            if key in self._ttl_dict:
                if time.time() > self._ttl_dict[key]:
                    # Expired, remove
                    del self._state_dict[key]
                    del self._ttl_dict[key]
                    logger.debug(f"State expired: {key}")
                    return None
            
            return self._state_dict.get(key)
    
    async def set_state(
        self,
        user_id: str,
        session_id: str,
        state_data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> None:
        """
        Set session state with TTL.
        
        Args:
            user_id: User ID
            session_id: Session ID
            state_data: State data dict
            ttl_seconds: TTL in seconds (default: 24h)
        
        LATENCY: <1ms
        """
        key = f"{user_id}:{session_id}"
        ttl = ttl_seconds or self._default_ttl
        
        async with self._lock:
            self._state_dict[key] = state_data
            self._ttl_dict[key] = time.time() + ttl
            
            logger.debug(f"State set: {key} (TTL: {ttl}s)")
    
    async def delete_state(self, user_id: str, session_id: str) -> bool:
        """
        Delete session state.
        
        Args:
            user_id: User ID
            session_id: Session ID
        
        Returns:
            True if deleted, False if not found
        """
        key = f"{user_id}:{session_id}"
        
        async with self._lock:
            if key in self._state_dict:
                del self._state_dict[key]
                del self._ttl_dict[key]
                logger.debug(f"State deleted: {key}")
                return True
            return False
    
    async def update_state(
        self,
        user_id: str,
        session_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update specific fields in session state.
        
        Args:
            user_id: User ID
            session_id: Session ID
            updates: Fields to update
        """
        key = f"{user_id}:{session_id}"
        
        async with self._lock:
            if key in self._state_dict:
                self._state_dict[key].update(updates)
                logger.debug(f"State updated: {key}")

# ================================================================================
# FUTURE EXTENSION (Phase 2+)
# ================================================================================

# TODO (Phase 2): Add Redis persistence
# TODO (Phase 2): Add distributed state (multi-region)
# TODO (Phase 2): Add state snapshots
# TODO (Phase 2): Add metrics (count, size, TTL)
# TODO (Phase 2): Add user preferences persistence
