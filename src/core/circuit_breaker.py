"""
================================================================================
CIRCUIT BREAKER - Fault Tolerance & Graceful Degradation
================================================================================

PURPOSE:
--------
Implement Circuit Breaker pattern to prevent cascading failures.

States:
  CLOSED   → Normal operation (requests flowing)
  OPEN     → Failure threshold reached (requests blocked)
  HALF_OPEN → Testing if service recovered (limited requests)

When to use:
  - Network requests (embedding API, vector DB)
  - External services
  - Unstable operations

Benefits:
  ✅ Fail fast instead of hanging
  ✅ Prevent cascading failures
  ✅ Automatic recovery when service healthy
  ✅ Better user experience (clear error vs timeout)

ARCHITECTURE:
--------------
     Pipeline node
           ↓
     CircuitBreakerManager
           ↓
     External service (API, DB, etc.)

CONFIG: config/settings/circuit_breaker.yaml
-------
    circuit_breaker:
      failure_threshold: 5          # Failures before OPEN
      success_threshold: 2          # Successes to close from HALF_OPEN
      timeout_seconds: 60           # Time in OPEN before HALF_OPEN
      check_interval: 10            # Check health interval

================================================================================
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
import asyncio
import logging

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Blocking requests
    HALF_OPEN = "half_open"    # Testing recovery


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = Field(5, ge=1, description="Failures before OPEN")
    success_threshold: int = Field(2, ge=1, description="Successes to close")
    timeout_seconds: int = Field(60, ge=1, description="Time before HALF_OPEN")
    check_interval: int = Field(10, ge=1, description="Health check interval")


class CircuitBreaker:
    """
    Circuit breaker for a single service/tool.
    
    Tracks failures and automatically stops sending requests
    when failure threshold exceeded.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the protected service
            config: Configuration
        """
        self.name = name
        self.config = config
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None
        
        logger.info(f"🔌 Circuit breaker created: {name}")

    def can_attempt(self) -> bool:
        """
        Check if we can attempt a request.
        
        Returns:
            bool: True if request should be attempted
        """
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.HALF_OPEN:
            return True
        
        # OPEN: Check if timeout elapsed
        if self.state == CircuitState.OPEN:
            if self.opened_at is None:
                return False
            
            elapsed = (datetime.utcnow() - self.opened_at).total_seconds()
            
            if elapsed >= self.config.timeout_seconds:
                logger.info(f"🔄 {self.name}: Transitioning OPEN → HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            
            return False
        
        return False

    def record_success(self) -> None:
        """Record a successful request."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                logger.info(f"✅ {self.name}: Transitioning HALF_OPEN → CLOSED")
                self.state = CircuitState.CLOSED
                self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            self.success_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.success_count = 0
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"⚠️  {self.name}: Failure threshold exceeded. "
                    f"Transitioning CLOSED → OPEN"
                )
                self.state = CircuitState.OPEN
                self.opened_at = datetime.utcnow()
        
        elif self.state == CircuitState.HALF_OPEN:
            logger.info(f"🔄 {self.name}: Failure in HALF_OPEN. Reopening.")
            self.state = CircuitState.OPEN
            self.opened_at = datetime.utcnow()

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
        }


class CircuitBreakerManager:
    """
    Manage circuit breakers for multiple services.
    
    Tracks all circuit breakers and provides centralized control.
    """

    def __init__(self, config: CircuitBreakerConfig):
        """
        Initialize manager.
        
        Args:
            config: Configuration for all breakers
        """
        self.config = config
        self.breakers: Dict[str, CircuitBreaker] = {}
        logger.info("🔌 Circuit breaker manager initialized")

    def get_breaker(self, name: str) -> CircuitBreaker:
        """
        Get or create a circuit breaker for a service.
        
        Args:
            name: Service name
            
        Returns:
            CircuitBreaker instance
        """
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, self.config)
        
        return self.breakers[name]

    async def call_with_breaker(
        self,
        service_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            service_name: Name of the service
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Circuit open or function failed
            
        Example:
            result = await breaker_manager.call_with_breaker(
                "embedding_api",
                embed_documents,
                texts=["doc1", "doc2"]
            )
        """
        from .exceptions import ProcessingError
        
        breaker = self.get_breaker(service_name)
        
        if not breaker.can_attempt():
            raise ProcessingError(
                f"Circuit breaker OPEN for {service_name}. "
                f"Service temporarily unavailable."
            )
        
        try:
            result = await func(*args, **kwargs)
            breaker.record_success()
            return result
        
        except Exception as e:
            breaker.record_failure()
            logger.error(f"❌ {service_name} failed: {str(e)}")
            raise

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }

    def reset_breaker(self, service_name: str) -> bool:
        """
        Manually reset a circuit breaker to CLOSED.
        
        Args:
            service_name: Service name
            
        Returns:
            bool: True if reset
        """
        if service_name not in self.breakers:
            return False
        
        breaker = self.breakers[service_name]
        logger.info(f"🔄 Manually resetting breaker: {service_name}")
        
        breaker.state = CircuitState.CLOSED
        breaker.failure_count = 0
        breaker.success_count = 0
        
        return True

    def reset_all(self) -> None:
        """Manually reset all circuit breakers."""
        logger.info("🔄 Resetting all circuit breakers")
        
        for breaker in self.breakers.values():
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
