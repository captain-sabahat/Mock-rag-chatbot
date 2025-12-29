"""No-op cache provider (fallback when Redis unavailable)."""

import logging
from typing import Any, Optional

from src.providers.cache.base import ICacheProvider

logger = logging.getLogger(__name__)


class NoOpCacheProvider(ICacheProvider):
    """No-operation cache provider (fallback when Redis is down)."""

    def __init__(self) -> None:
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize (no-op)."""
        self.initialized = True
        logger.info("✓ NoOpCacheProvider initialized (caching disabled)")

    async def get(self, key: str) -> Optional[Any]:
        """Always returns None (no caching)."""
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """No-op set."""
        pass

    async def delete(self, key: str) -> None:
        """No-op delete."""
        pass

    async def clear(self) -> None:
        """No-op clear."""
        pass

    async def shutdown(self) -> None:
        """Shutdown (no-op)."""
        self.initialized = False
        logger.info("NoOpCacheProvider shutdown")


default_provider = NoOpCacheProvider()
