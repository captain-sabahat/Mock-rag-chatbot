# Resilience pattern
"""
================================================================================
FILE: src/core/circuit_breaker.py
================================================================================

PURPOSE:
    Implements circuit breaker resilience pattern. Prevents cascading failures
    by stopping repeated calls to failing services. Enables graceful degradation
    when backends are temporarily down.

WORKFLOW:
    1. Track failure count per service
    2. When failures exceed threshold → circuit OPENS
    3. While OPEN → fail immediately (fast-fail, don't call backend)
    4. After timeout → transition to HALF_OPEN (test recovery)
    5. If test succeeds → CLOSED (back to normal)
    6. If test fails → OPEN again (retry timeout)

IMPORTS:
    - asyncio: Async locking, timeout management
    - logging: Incident logging
    - time: Timestamp tracking
    - config.constants: Thresholds from configuration

INPUTS:
    - Function to protect (async function)
    - Arguments for function
    - Failure threshold (default: 3)
    - Recovery timeout (default: 60s)

OUTPUTS:
    - Function result (if successful)
    - CircuitBreakerOpenError (if circuit is open)
    - Logged incidents (failures, state transitions)

CIRCUIT BREAKER STATES:
    - CLOSED: Normal operation, call backend
    - OPEN: Too many failures, fail immediately
    - HALF_OPEN: Testing if service recovered, try once

STATE TRANSITIONS:
    CLOSED → OPEN: When failure_count >= threshold
    OPEN → HALF_OPEN: After recovery_timeout expires
    HALF_OPEN → CLOSED: If test call succeeds
    HALF_OPEN → OPEN: If test call fails

RESILIENCE STRATEGY:
    - Fail fast (don't wait for timeout if circuit open)
    - Allow graceful degradation (skip failed operation)
    - Auto-recovery (periodically test if service is back)
    - Prevents thundering herd (all clients back off together)

KEY FACTS:
    - Prevents repeated failures from cascading
    - Uses asyncio.Lock (non-blocking)
    - Tracks state across multiple requests
    - Auto-resets after timeout (no manual intervention)
    - Per-handler instance (isolated failure tracking)

FUTURE SCOPE (Phase 2+):
    - Add metrics collection (failure count, state transitions)
    - Add adaptive timeout (increase backoff exponentially)
    - Add jitter (randomize backoff to prevent thundering herd)
    - Add fallback service support
    - Add circuit breaker dashboard
    - Add alerts on state changes
    - Add per-handler configuration
    - Add manual override (force open/closed)

TESTING ENVIRONMENT:
    - Mock circuit breaker in tests
    - Test state transitions
    - Verify fast-fail behavior
    - Test recovery sequence

PRODUCTION DEPLOYMENT:
    - Monitor state transitions (alert on OPEN)
    - Track metrics (error rates, recovery success)
    - Tune thresholds per backend
    - Enable graceful degradation
"""

# ================================================================================
# IMPORTS
# ================================================================================

import asyncio
import logging
import time
from typing import Callable, Any

from src.config.constants import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS
)
from .exceptions import CircuitBreakerOpenError

logger = logging.getLogger(__name__)

# ================================================================================
# CIRCUIT BREAKER CLASS
# ================================================================================

class CircuitBreaker:
    """
    Circuit breaker for resilience pattern.
    
    Prevents cascading failures by failing fast when a service is down.
    Automatically tries to recover after timeout.
    """
    
    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening (default: 3)
            recovery_timeout: Seconds to wait before trying recovery (default: 60)
        """
        self._lock = asyncio.Lock()
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        
        # State tracking
        self._state = "CLOSED"  # CLOSED, OPEN, or HALF_OPEN
        self._failure_count = 0
        self._last_failure_time: float = None
        self._last_state_change_time: float = None
        
        logger.info(
            f"CircuitBreaker initialized: threshold={failure_threshold}, "
            f"timeout={recovery_timeout}s"
        )
    
    async def protect(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            fn: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpenError: If circuit is OPEN (fail fast)
        """
        async with self._lock:
            # Check if circuit should transition to HALF_OPEN
            if self._state == "OPEN":
                time_since_open = time.time() - self._last_state_change_time
                if time_since_open >= self._recovery_timeout:
                    # Try recovery
                    self._state = "HALF_OPEN"
                    self._failure_count = 0
                    logger.info("CircuitBreaker transitioned OPEN → HALF_OPEN (testing recovery)")
                else:
                    # Still in backoff period
                    time_remaining = self._recovery_timeout - time_since_open
                    logger.warning(
                        f"CircuitBreaker OPEN: fail fast. Retry in {time_remaining:.1f}s"
                    )
                    raise CircuitBreakerOpenError(
                        f"Service unavailable (circuit breaker open for {time_since_open:.1f}s)"
                    )
        
        # Execute function (outside lock, non-blocking)
        try:
            result = await fn(*args, **kwargs)
            
            # Success: record it
            async with self._lock:
                if self._state == "HALF_OPEN":
                    # Recovery successful
                    self._state = "CLOSED"
                    self._failure_count = 0
                    logger.info("CircuitBreaker transitioned HALF_OPEN → CLOSED (recovered)")
                elif self._state == "CLOSED":
                    # Normal success
                    self._failure_count = 0
            
            return result
        
        except Exception as e:
            # Failure: record it
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                logger.error(
                    f"CircuitBreaker: failure {self._failure_count}/{self._failure_threshold}: {str(e)}"
                )
                
                # Check if threshold exceeded
                if self._failure_count >= self._failure_threshold:
                    if self._state != "OPEN":
                        self._state = "OPEN"
                        self._last_state_change_time = time.time()
                        logger.error(
                            f"CircuitBreaker transitioned to OPEN after {self._failure_count} failures"
                        )
            
            # Re-raise exception (let caller handle)
            raise
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        return self._state
    
    def get_failure_count(self) -> int:
        """Get current failure count"""
        return self._failure_count
    
    async def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED"""
        async with self._lock:
            old_state = self._state
            self._state = "CLOSED"
            self._failure_count = 0
            self._last_failure_time = None
            logger.info(f"CircuitBreaker manually reset from {old_state} to CLOSED")

# ================================================================================
# FUTURE EXTENSION (Phase 2+)
# ================================================================================

# TODO (Phase 2): Add Prometheus metrics (state transitions, failure count)
# TODO (Phase 2): Add adaptive backoff (exponential timeout increase)
# TODO (Phase 2): Add jitter to prevent thundering herd
# TODO (Phase 2): Add fallback service support
# TODO (Phase 2): Add per-handler threshold tuning
# TODO (Phase 2): Add circuit breaker dashboard/monitoring
