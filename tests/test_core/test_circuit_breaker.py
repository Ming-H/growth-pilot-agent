"""Tests for src.core.circuit_breaker - Circuit Breaker pattern."""
from __future__ import annotations

import asyncio
import time

import pytest

from src.core.circuit_breaker import CircuitBreaker, CircuitState


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestCircuitBreakerInitialState:
    """Tests for circuit breaker initial state."""

    def test_starts_closed(self):
        """A new circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_starts_with_zero_failures(self):
        """Failure count starts at zero."""
        cb = CircuitBreaker()
        assert cb.failure_count == 0

    def test_custom_threshold(self):
        """Custom failure_threshold is respected."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30


# ---------------------------------------------------------------------------
# CLOSED -> OPEN transition
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpenTransition:
    """Tests for transitioning from CLOSED to OPEN."""

    @pytest.mark.asyncio
    async def test_transitions_to_open_after_threshold_failures(self):
        """Circuit opens after reaching the failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        @cb
        async def failing():
            raise RuntimeError("fail")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await failing()

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    @pytest.mark.asyncio
    async def test_stays_closed_below_threshold(self):
        """Circuit stays CLOSED when failures are below threshold."""
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

        @cb
        async def failing():
            raise RuntimeError("fail")

        for _ in range(4):
            with pytest.raises(RuntimeError):
                await failing()

        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_exact_threshold_opens(self):
        """Circuit opens at exactly the threshold."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)

        @cb
        async def failing():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            await failing()

        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# OPEN state - reject calls
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpenRejects:
    """Tests for call rejection when circuit is OPEN."""

    @pytest.mark.asyncio
    async def test_rejects_calls_when_open(self):
        """Calls are rejected immediately when the circuit is OPEN."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)

        @cb
        async def failing():
            raise RuntimeError("fail")

        # Trigger open
        with pytest.raises(RuntimeError):
            await failing()
        assert cb.state == CircuitState.OPEN

        # Subsequent call should be rejected by circuit breaker, not the function
        with pytest.raises(RuntimeError, match="Circuit breaker OPEN"):
            await failing()

    @pytest.mark.asyncio
    async def test_rejection_does_not_invoke_function(self):
        """When OPEN, the wrapped function is never called."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        call_count = 0

        @cb
        async def tracked():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await tracked()

        # This rejection should NOT increment call_count
        with pytest.raises(RuntimeError, match="Circuit breaker OPEN"):
            await tracked()

        assert call_count == 1


# ---------------------------------------------------------------------------
# OPEN -> HALF_OPEN transition
# ---------------------------------------------------------------------------


class TestCircuitBreakerHalfOpenTransition:
    """Tests for transitioning from OPEN to HALF_OPEN."""

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery_timeout elapses."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        @cb
        async def failing():
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await failing()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should transition to HALF_OPEN and attempt the function
        with pytest.raises(RuntimeError):
            await failing()

        # State should be OPEN again (failure in HALF_OPEN transitions back)
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_allows_one_probe(self):
        """In HALF_OPEN, a limited number of probe requests are allowed."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1, half_open_max=1)

        call_count = 0

        @cb
        async def tracked():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await tracked()

        # Wait for recovery
        await asyncio.sleep(0.15)

        # First probe goes through
        with pytest.raises(RuntimeError):
            await tracked()
        assert call_count == 2


# ---------------------------------------------------------------------------
# HALF_OPEN -> CLOSED on success
# ---------------------------------------------------------------------------


class TestCircuitBreakerRecovery:
    """Tests for recovery from HALF_OPEN to CLOSED."""

    @pytest.mark.asyncio
    async def test_success_in_half_open_closes_circuit(self):
        """A successful call in HALF_OPEN transitions back to CLOSED."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        should_fail = True

        @cb
        async def conditional():
            if should_fail:
                raise RuntimeError("fail")
            return "ok"

        # Open the circuit
        with pytest.raises(RuntimeError):
            await conditional()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Fix the function
        should_fail = False

        # Successful call in HALF_OPEN should close the circuit
        result = await conditional()
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        """A successful call resets the failure counter to 0."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        @cb
        async def sometimes_fails():
            return "ok"

        # Should work fine and reset any prior count
        await sometimes_fails()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# HALF_OPEN -> OPEN on failure
# ---------------------------------------------------------------------------


class TestCircuitBreakerHalfOpenFailure:
    """Tests for HALF_OPEN -> OPEN transition on failure."""

    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens(self):
        """A failure in HALF_OPEN transitions back to OPEN."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        @cb
        async def always_fails():
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await always_fails()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Failure in HALF_OPEN reopens circuit
        with pytest.raises(RuntimeError):
            await always_fails()
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_limit_enforced(self):
        """HALF_OPEN enforces the half_open_max limit."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1, half_open_max=1)

        @cb
        async def always_fails():
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await always_fails()

        # Wait for recovery
        await asyncio.sleep(0.15)

        # First call is allowed (half_open_count becomes 1)
        with pytest.raises(RuntimeError):
            await always_fails()

        # Circuit is OPEN again (failure reopened it)
        assert cb.state == CircuitState.OPEN

        # Wait for recovery again
        await asyncio.sleep(0.15)

        # Now one more probe should be allowed
        with pytest.raises(RuntimeError):
            await always_fails()


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestCircuitBreakerLifecycle:
    """Integration test: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Complete lifecycle: closed -> open -> half_open -> closed."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        should_fail = True

        @cb
        async def service():
            if should_fail:
                raise RuntimeError("service down")
            return "ok"

        # CLOSED: normal operation, accumulating failures
        assert cb.state == CircuitState.CLOSED

        with pytest.raises(RuntimeError):
            await service()
        assert cb.state == CircuitState.CLOSED  # Not yet at threshold

        with pytest.raises(RuntimeError):
            await service()
        assert cb.state == CircuitState.OPEN  # Threshold reached

        # OPEN: calls rejected
        with pytest.raises(RuntimeError, match="Circuit breaker OPEN"):
            await service()

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Fix the service
        should_fail = False

        # HALF_OPEN -> success -> CLOSED
        result = await service()
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

        # Normal operation resumes
        result = await service()
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED
