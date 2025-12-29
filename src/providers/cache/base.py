from abc import ABC, abstractmethod
from typing import Any, Optional


class ICacheProvider(ABC):
    """Abstract base class for all cache providers."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the cache provider."""
        pass

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a value from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries in the cache."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown cache provider and release resources."""
        pass
