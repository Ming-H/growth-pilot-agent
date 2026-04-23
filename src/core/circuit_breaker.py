"""Circuit Breaker pattern for protecting async call sites.

Usage as a decorator::

    from src.core.circuit_breaker import CircuitBreaker

    @CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    async def call_external_api(...):
        ...
"""

from __future__ import annotations

import time
from enum import Enum
from functools import wraps
from typing import Any, Callable


class CircuitState(Enum):
    """Possible states of a circuit breaker."""

    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Async decorator that implements the Circuit Breaker resilience pattern.

    Parameters
    ----------
    failure_threshold:
        Number of consecutive failures before the circuit opens.
    recovery_timeout:
        Seconds to wait in OPEN state before transitioning to HALF_OPEN.
    half_open_max:
        Maximum number of probe requests allowed through in HALF_OPEN state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60,
        half_open_max: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: float = 0
        self.half_open_count = 0

    # ------------------------------------------------------------------
    # Decorator interface
    # ------------------------------------------------------------------

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap *func* so that calls are guarded by the circuit breaker."""

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # --- OPEN state ---
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_count = 0
                else:
                    raise RuntimeError(
                        f"Circuit breaker OPEN for {func.__name__}"
                    )

            # --- HALF_OPEN state ---
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_count >= self.half_open_max:
                    raise RuntimeError(
                        f"Circuit breaker HALF_OPEN limit for {func.__name__}"
                    )
                self.half_open_count += 1

            # --- Execute the guarded call ---
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except Exception:
                self._on_failure()
                raise

        return wrapper

    # ------------------------------------------------------------------
    # Internal state transitions
    # ------------------------------------------------------------------

    def _on_success(self) -> None:
        """Reset failure counter and close the circuit."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        """Increment failure counter; open the circuit if threshold is met."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
