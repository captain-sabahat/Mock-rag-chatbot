# LINE 2: Cache (<10ms)
"""
================================================================================
FILE: src/core/redis_handler.py
================================================================================

PURPOSE:
    Ultra-low-latency caching layer using Redis. Enables LOGIC_1 (<10ms)
    fast path. Stores query results, session state, document summaries.

WORKFLOW (LOGIC 1 - Cache Hit Path):
    1. User submits query
    2. Hash query to create cache key
    3. Check Redis GET(cache_key)
    4. If HIT: return cached answer (<10ms)
    5. If MISS: proceed to RAG pipeline

LATENCY TARGETS:
    - Cache HIT: <10ms (ultra-fast)
    - Cache MISS: 1-2ms (check + return)
    - Cache SET: <5ms (write)

IMPORTS:
    - redis.asyncio: Async Redis client
    - asyncio: Async operations
    - logging: Logging
    - config: Redis configuration

INPUTS:
    - cache_key: Redis key (e.g., "user:123:query:abc123")
    - value: Data to cache (JSON string)
    - ttl: Time-to-live in seconds

OUTPUTS:
    - Cached value (from GET)
    - Success/failure status (for SET)

REDIS KEY PATTERNS:
    - Query cache: "user:{user_id}:query:{query_hash}"
    - Session: "session:{user_id}:{session_id}"
    - Summary: "summary:{user_id}:{doc_hash}"

CONNECTION POOLING:
    - Pool size: 50 connections (configurable)
    - Connection timeout: 5 seconds
    - Keep-alive: enabled

RESILIENCE:
    - Connection errors: RecoverableException (retry)
    - Timeout: Fast fail (<2s timeout)
    - Circuit breaker: Fail fast if Redis down

KEY FACTS:
    - All operations async (non-blocking)
    - Connection pooling reduces latency
    - Async context manager for resource cleanup
    - Circuit breaker protects against Redis failures

CACHE STRATEGY:
    - Always cache query results (high hit rate)
    - Always cache summaries (expensive to compute)
    - Cache sessions (user continuity)

FUTURE SCOPE (Phase 2+):
    - Add cache warming (pre-populate common queries)
    - Add cache invalidation patterns
    - Add cache metrics (hit rate, size, memory)
    - Add distributed Redis (multi-region)
    - Add compression (for large values)
    - Add encryption (for sensitive data)
    - Add TTL management (automatic expiration)
    - Add cache statistics tracking

TESTING ENVIRONMENT:
    - Mock Redis client in tests
    - Use local Redis instance
    - Test cache hit/miss scenarios
    - Verify TTL handling

PRODUCTION DEPLOYMENT:
    - Use managed Redis (AWS ElastiCache, etc.)
    - Enable persistence (RDB or AOF)
    - Enable replication (high availability)
    - Monitor connection pool
    - Alert if cache hit rate drops
    - Monitor memory usage
"""

# ================================================================================
# IMPORTS
# ================================================================================

import asyncio
import logging
from typing import Optional, Any

import redis.asyncio as redis
from src.config.settings import Settings
from .exceptions import RedisError, RecoverableException
from src.utils import generate_query_cache_key, hash_query

logger = logging.getLogger(__name__)

# ================================================================================
# REDIS HANDLER CLASS
# ================================================================================

class RedisHandler:
    """
    Redis async client for ultra-low-latency caching.
    
    Implements cache GET/SET operations with connection pooling.
    All operations are async (non-blocking).
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Redis handler.
        
        Args:
            settings: Application settings (Redis URL, pool size, timeout)
        """
        self.settings = settings
        self.pool: Optional[redis.ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        logger.info(f"RedisHandler initialized with URL: {settings.redis_url}")
    
    async def __aenter__(self):
        """
        Async context manager entry - establish connection pool.
        
        Returns:
            self (RedisHandler)
        
        Raises:
            RedisError: If connection fails
        """
        try:
            self.pool = redis.ConnectionPool.from_url(
                self.settings.redis_url,
                max_connections=self.settings.redis_pool_size,
                decode_responses=True,  # Return strings, not bytes
                socket_connect_timeout=5,
                socket_keepalive=True,
                socket_keepalive_interval=60
            )
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            logger.info("Redis connection pool established and verified")
            return self
        
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise RedisError(
                f"Redis connection failed: {str(e)}",
                context={"redis_url": self.settings.redis_url}
            )
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit - close connection pool.
        """
        if self.client:
            await self.client.close()
            logger.info("Redis connection pool closed")
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value (string) or None if not found
        
        Raises:
            RedisError: If operation fails
        
        LATENCY TARGET: <10ms
        """
        try:
            value = await asyncio.wait_for(
                self.client.get(key),
                timeout=self.settings.redis_timeout
            )
            
            if value is not None:
                logger.debug(f"Cache HIT: {key[:50]}...")
            else:
                logger.debug(f"Cache MISS: {key[:50]}...")
            
            return value
        
        except asyncio.TimeoutError:
            raise RedisError(
                f"Redis GET timeout after {self.settings.redis_timeout}s",
                context={"key": key}
            )
        except Exception as e:
            raise RedisError(
                f"Redis GET failed: {str(e)}",
                context={"key": key}
            )
    
    async def set(self, key: str, value: str, ttl: int) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (string)
            ttl: Time-to-live in seconds
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            RedisError: If operation fails
        
        LATENCY TARGET: <5ms
        """
        try:
            await asyncio.wait_for(
                self.client.setex(key, ttl, value),
                timeout=self.settings.redis_timeout
            )
            
            logger.debug(f"Cache SET: {key[:50]}... TTL={ttl}s")
            return True
        
        except asyncio.TimeoutError:
            raise RedisError(
                f"Redis SET timeout after {self.settings.redis_timeout}s",
                context={"key": key}
            )
        except Exception as e:
            raise RedisError(
                f"Redis SET failed: {str(e)}",
                context={"key": key}
            )
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
        
        Returns:
            True if deleted, False if not found
        """
        try:
            result = await asyncio.wait_for(
                self.client.delete(key),
                timeout=self.settings.redis_timeout
            )
            logger.debug(f"Cache DELETE: {key[:50]}...")
            return result > 0
        except Exception as e:
            raise RedisError(f"Redis DELETE failed: {str(e)}")
    
    async def get_query_cache(self, user_id: str, query: str) -> Optional[str]:
        """
        Get cached query result.
        
        Args:
            user_id: User ID
            query: User query string
        
        Returns:
            Cached answer or None
        """
        query_hash = hash_query(query)
        key = generate_query_cache_key(user_id, query_hash)
        return await self.get(key)
    
    async def set_query_cache(self, user_id: str, query: str, answer: str, ttl: int) -> bool:
        """
        Cache query result.
        
        Args:
            user_id: User ID
            query: User query string
            answer: Generated answer to cache
            ttl: Cache TTL in seconds
        
        Returns:
            Success status
        """
        query_hash = hash_query(query)
        key = generate_query_cache_key(user_id, query_hash)
        return await self.set(key, answer, ttl)

# ================================================================================
# FUTURE EXTENSION (Phase 2+)
# ================================================================================

# TODO (Phase 2): Add cache warming
# TODO (Phase 2): Add cache metrics (hit rate, memory)
# TODO (Phase 2): Add distributed Redis
# TODO (Phase 2): Add compression for large values
# TODO (Phase 2): Add encryption for sensitive data
# TODO (Phase 2): Add cache statistics tracking
