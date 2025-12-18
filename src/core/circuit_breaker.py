# ============================================================================
# CIRCUIT BREAKER - Fault Tolerance & Graceful Degradation (v2.2)
# ============================================================================
"""
CRITICAL FIX (v2.2):
1. Circuit breaker tracks which node opened it
2. Reason for opening is captured and accessible
3. Routes can query CB state for ingestions
4. Orchestrator passes CB state to monitoring system

================================================================================
CIRCUIT BREAKER - Fault Tolerance & Graceful Degradation
================================================================================

PURPOSE:
--------
Implement Circuit Breaker pattern to prevent cascading failures.
Integrates with pipeline orchestrator to enforce conditions A-D.

States:
CLOSED → Normal operation (requests flowing)
OPEN → Failure threshold reached (requests blocked)
HALF_OPEN → Testing if service recovered (limited requests)

Circuit Breaker Conditions (A-D):
A: CRITICAL Exception → OPEN
B: No Output or Invalid Output → OPEN
C: Data Transfer Failure → OPEN
D: Time Gap > 30s → OPEN

When to use:
- Network requests (embedding API, vector DB)
- External services
- Unstable operations

Benefits:
✅ Fail fast instead of hanging
✅ Prevent cascading failures
✅ Automatic recovery when service healthy
✅ Better user experience (clear error vs timeout)

CONFIG: config/settings/circuit_breaker.yaml
-------
circuit_breaker:
  failure_threshold: 5 # Failures before OPEN
  success_threshold: 2 # Successes to close from HALF_OPEN
  timeout_seconds: 60 # Time in OPEN before HALF_OPEN
  check_interval: 10 # Check health interval

================================================================================
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
import asyncio
import logging

logger = logging.getLogger(__name__)

# ================================================================================
# CIRCUIT BREAKER STATE
# ================================================================================

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Blocking requests
    HALF_OPEN = "half_open"    # Testing recovery


# ================================================================================
# CIRCUIT BREAKER CONFIGURATION
# ================================================================================

class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    failure_threshold: int = Field(5, ge=1, description="Failures before OPEN")
    success_threshold: int = Field(2, ge=1, description="Successes to close")
    timeout_seconds: int = Field(60, ge=1, description="Time before HALF_OPEN")
    check_interval: int = Field(10, ge=1, description="Health check interval")


# ================================================================================
# SINGLE CIRCUIT BREAKER
# ================================================================================

class CircuitBreaker:
    """
    Circuit breaker for a single service/tool.
    
    ✅ ENHANCED (v2.2): Tracks which node opened it and why
    
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
        self.last_check_time: Optional[datetime] = None
        
        # ✅ NEW (v2.2): Track which node/operation opened the CB and why
        self.opened_node: Optional[str] = None
        self.opened_reason: Optional[str] = None
        
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
                # ✅ NEW (v2.2): Clear opened node/reason when closing
                self.opened_node = None
                self.opened_reason = None

        elif self.state == CircuitState.CLOSED:
            self.success_count = 0

    def record_failure(self, reason: str = "Unknown", node: Optional[str] = None) -> None:
        """
        ✅ ENHANCED (v2.2): Record a failed request with node information.

        Args:
            reason: Reason for failure (for logging)
            node: Name of the node where failure occurred (NEW)
        """
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        self.success_count = 0

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"⚠️ {self.name}: Failure threshold exceeded ({reason}). "
                    f"Transitioning CLOSED → OPEN"
                )
                self.state = CircuitState.OPEN
                self.opened_at = datetime.utcnow()
                # ✅ NEW (v2.2): Track which node opened it
                self.opened_node = node
                self.opened_reason = reason

        elif self.state == CircuitState.HALF_OPEN:
            logger.info(
                f"🔄 {self.name}: Failure in HALF_OPEN ({reason}). Reopening."
            )
            self.state = CircuitState.OPEN
            self.opened_at = datetime.utcnow()
            # ✅ NEW (v2.2): Track which node opened it
            self.opened_node = node
            self.opened_reason = reason

    # ✅ NEW (v2.2): Get which node opened the circuit breaker
    def get_opened_node(self) -> Optional[str]:
        """
        Get the node that opened the circuit breaker.
        
        Returns:
            str: Node name (e.g., "embedding", "vectordb") or None if CLOSED
        """
        return self.opened_node

    # ✅ NEW (v2.2): Get the reason the circuit breaker opened
    def get_opened_reason(self) -> Optional[str]:
        """
        Get the reason the circuit breaker opened.
        
        Returns:
            str: Reason string or None if CLOSED
        """
        return self.opened_reason

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            # ✅ NEW (v2.2): Include opened node and reason in status
            "opened_node": self.opened_node,
            "opened_reason": self.opened_reason,
        }


# ================================================================================
# CIRCUIT BREAKER MANAGER (for orchestrator integration)
# ================================================================================

class CircuitBreakerManager:
    """
    ✅ ENHANCED (v2.2): Manage circuit breakers with state queries
    
    Manage circuit breakers for multiple services.
    Tracks all circuit breakers and provides centralized control.
    Integrates with orchestrator for pipeline-level decisions.
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

    # ✅ NEW (v2.2): Check if any breaker is OPEN
    def is_any_open(self) -> bool:
        """
        Check if ANY circuit breaker is OPEN.
        
        Returns:
            bool: True if any breaker is OPEN
        """
        return any(b.state == CircuitState.OPEN for b in self.breakers.values())

    # ✅ NEW (v2.2): Get which breaker(s) are OPEN
    def get_open_breakers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all OPEN circuit breakers with details.
        
        Returns:
            Dict of breaker_name -> {state, reason, opened_node}
        """
        return {
            name: {
                "state": breaker.state.value,
                "opened_node": breaker.get_opened_node(),
                "opened_reason": breaker.get_opened_reason(),
                "opened_at": breaker.opened_at.isoformat() if breaker.opened_at else None,
            }
            for name, breaker in self.breakers.items()
            if breaker.state == CircuitState.OPEN
        }

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

        Usage:
            result = await breaker_manager.call_with_breaker(
                "embedding_api",
                embed_documents,
                texts=["doc1", "doc2"]
            )
        """
        from src.core.exceptions import ProcessingError

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
            breaker.record_failure(reason=str(e), node=service_name)
            logger.error(f"❌ {service_name} failed: {str(e)}")
            raise

    def record_failure_for_node(
        self,
        node_name: str,
        reason: str,
        exception_severity: str = "CRITICAL"
    ) -> bool:
        """
        ✅ ENHANCED (v2.2): Record failure for a node with node tracking.
        
        Implements Condition A: CRITICAL exception → OPEN

        Args:
            node_name: Name of the pipeline node
            reason: Reason for failure
            exception_severity: Severity level (CRITICAL, WARNING, INFO)

        Returns:
            bool: True if circuit breaker was opened

        Usage (from orchestrator):
            from src.core.exceptions import should_trigger_circuit_breaker

            if should_trigger_circuit_breaker(exception_type):
                cb_manager.record_failure_for_node(
                    "ingestion",
                    "PDF parsing failed",
                    "CRITICAL"
                )
        """
        breaker = self.get_breaker(node_name)

        # Only CRITICAL exceptions trigger circuit breaker
        if exception_severity == "CRITICAL":
            breaker.record_failure(reason=reason, node=node_name)
            return breaker.state == CircuitState.OPEN

        return False

    def check_condition_d(
        self,
        node_name: str,
        time_gap_seconds: float,
        threshold_seconds: float = 30.0
    ) -> bool:
        """
        Check Condition D: Time Gap > 30s

        Args:
            node_name: Name of the node
            time_gap_seconds: Time gap between node executions
            threshold_seconds: Threshold (default 30s)

        Returns:
            bool: True if condition D triggered (should OPEN)

        Usage (from orchestrator):
            if orchestrator.cb_manager.check_condition_d(
                "embedding",
                time_gap_ms / 1000,
                threshold=30
            ):
                # Condition D triggered - open circuit breaker
                pass
        """
        breaker = self.get_breaker(node_name)

        if time_gap_seconds > threshold_seconds:
            logger.warning(
                f"⚠️ Condition D (time gap > {threshold_seconds}s) triggered "
                f"for node '{node_name}': gap={time_gap_seconds:.2f}s"
            )
            breaker.record_failure(
                reason=f"Time gap exceeded: {time_gap_seconds:.2f}s",
                node=node_name
            )
            return breaker.state == CircuitState.OPEN

        return False

    def get_all_status(self) -> Dict[str, Any]:
        """
        Get status of all breakers.
        
        ✅ ENHANCED (v2.2): Includes opened_node and opened_reason
        """
        return {
            "state": self.get_circuit_breaker_state(),
            "breakers": {
                name: breaker.get_status()
                for name, breaker in self.breakers.items()
            },
        }

    def get_circuit_breaker_state(self) -> str:
        """
        Get overall circuit breaker state.

        Returns:
            "CLOSED" if all breakers closed
            "HALF_OPEN" if any breaker is half-open
            "OPEN" if any breaker is open
        """
        states = [breaker.state for breaker in self.breakers.values()]

        if not states:
            return "CLOSED"

        if CircuitState.OPEN in states:
            return "OPEN"

        if CircuitState.HALF_OPEN in states:
            return "HALF_OPEN"

        return "CLOSED"

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
        # ✅ NEW (v2.2): Clear opened info on manual reset
        breaker.opened_node = None
        breaker.opened_reason = None

        return True

    def reset_all(self) -> None:
        """Manually reset all circuit breakers."""
        logger.info("🔄 Resetting all circuit breakers")
        for breaker in self.breakers.values():
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            # ✅ NEW (v2.2): Clear opened info
            breaker.opened_node = None
            breaker.opened_reason = None
