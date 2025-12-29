"""
FILE: src/providers/cache/redis.py

Redis cache provider implementation.
"""

import json
import logging
from typing import Any, Optional

from src.providers.cache.base import ICacheProvider
from src.core.redis_handler import RedisHandler
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class RedisCacheProvider(ICacheProvider):
    """Redis cache provider implementation."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_handler: Optional[RedisHandler] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.redis_handler = redis_handler
        self.client = None
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            if self.redis_handler is None:
                # Instantiate RedisHandler with settings
                self.redis_handler = RedisHandler(self.settings)

            # Get client directly
            self.client = self.redis_handler.client

            if self.client:
                await self.client.ping()
                self.initialized = True
                logger.info("✓ RedisCacheProvider initialized (Redis connected)")
            else:
                logger.warning("Redis client not available")
                self.initialized = False

        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {str(e)}")
            self.initialized = False

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not self.initialized or not self.client:
            return None

        try:
            value = await self.client.get(key)
            if value is None:
                return None

            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.warning(f"Error getting cache key '{key}': {str(e)}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in Redis."""
        if not self.initialized or not self.client:
            return

        try:
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError):
                serialized = str(value)

            if ttl is not None:
                await self.client.setex(key, ttl, serialized)
            else:
                await self.client.set(key, serialized)
        except Exception as e:
            logger.warning(f"Error setting cache key '{key}': {str(e)}")

    async def delete(self, key: str) -> None:
        """Delete from Redis."""
        if not self.initialized or not self.client:
            return

        try:
            await self.client.delete(key)
        except Exception as e:
            logger.warning(f"Error deleting cache key '{key}': {str(e)}")

    async def clear(self) -> None:
        """Clear all Redis cache."""
        if not self.initialized or not self.client:
            return

        try:
            await self.client.flushdb()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing cache: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown Redis connection."""
        try:
            if self.redis_handler:
                await self.redis_handler.shutdown()
            self.initialized = False
            logger.info("RedisCacheProvider shutdown")
        except Exception as e:
            logger.warning(f"Error shutting down RedisCacheProvider: {str(e)}")

    # ------------------------------------------------------------------
    # Query-cache helpers to match rag.py expectations
    # ------------------------------------------------------------------

    async def set_query_cache(
        self,
        user_id: str,
        query: str,
        answer: str,
        ttl: int,
    ) -> bool:
        """
        Cache query result for a given user.

        Delegates to RedisHandler.set_query_cache, which handles key generation.
        """
        if not self.redis_handler:
            logger.warning("RedisHandler not initialized; cannot set_query_cache")
            return False
        try:
            return await self.redis_handler.set_query_cache(
                user_id=user_id,
                query=query,
                answer=answer,
                ttl=ttl,
            )
        except Exception as e:
            logger.warning(f"Error in set_query_cache for user '{user_id}': {str(e)}")
            return False

    async def get_query_cache(self, user_id: str, query: str) -> Optional[str]:
        """
        Get cached query result for a given user.

        Delegates to RedisHandler.get_query_cache, which handles key generation.
        """
        if not self.redis_handler:
            logger.warning("RedisHandler not initialized; cannot get_query_cache")
            return None
        try:
            return await self.redis_handler.get_query_cache(
                user_id=user_id,
                query=query,
            )
        except Exception as e:
            logger.warning(f"Error in get_query_cache for user '{user_id}': {str(e)}")
            return None


# Singleton instance used by ServiceContainer
default_provider = RedisCacheProvider()
