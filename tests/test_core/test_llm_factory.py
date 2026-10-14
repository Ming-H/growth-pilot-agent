"""Tests for src.core.llm_factory — resilience wiring and shared breakers."""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.core.llm_factory import (
    CircuitBreaker,
    CircuitState,
    ResilientLLM,
    _get_shared_breaker,
    _reset_breaker_registry,
    create_resilient_llm,
)


def _make_settings(**overrides) -> Settings:
    """Build Settings with safe test defaults; no real API calls happen."""
    base = dict(
        llm_provider="openai",
        llm_model="test-model",
        llm_api_key="sk-test",
        llm_base_url="https://primary.example.com",
        fallback_enabled=True,
        circuit_breaker_threshold=3,
        circuit_breaker_recovery_seconds=60.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _clean_breaker_registry():
    """Isolate the shared breaker registry between tests."""
    _reset_breaker_registry()
    yield
    _reset_breaker_registry()


class TestCircuitBreaker:
    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        # Only 1 consecutive failure after the reset — still closed
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self, monkeypatch):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Simulate time passing beyond the recovery window
        cb._last_failure_time -= 20.0
        assert cb.state == CircuitState.HALF_OPEN


class TestSharedBreakerRegistry:
    def test_same_endpoint_shares_breaker(self):
        a = _get_shared_breaker("openai", "m", "https://x", 3, 60.0)
        b = _get_shared_breaker("openai", "m", "https://x", 3, 60.0)
        assert a is b

    def test_different_endpoints_get_distinct_breakers(self):
        a = _get_shared_breaker("openai", "m", "https://x", 3, 60.0)
        b = _get_shared_breaker("deepseek", "m2", "https://y", 3, 60.0)
        assert a is not b


class TestCreateResilientLLM:
    def test_self_fallback_is_rejected(self):
        """Fallback identical to primary (provider+model+endpoint) is disabled."""
        s = _make_settings(
            fallback_provider="openai",
            fallback_model="test-model",
            fallback_base_url="https://primary.example.com",
        )
        result = create_resilient_llm(settings=s)
        assert not isinstance(result, ResilientLLM)

    def test_heterogeneous_fallback_returns_resilient_proxy(self):
        s = _make_settings(
            fallback_provider="deepseek",
            fallback_model="deepseek-chat",
            fallback_api_key="sk-fallback",
        )
        result = create_resilient_llm(settings=s)
        assert isinstance(result, ResilientLLM)

    def test_resilient_instances_share_breaker_state(self):
        """Two call sites targeting the same endpoint share one breaker."""
        s = _make_settings(
            fallback_provider="deepseek",
            fallback_model="deepseek-chat",
            fallback_api_key="sk-fallback",
        )
        a = create_resilient_llm(settings=s, agent_name="plan_node")
        b = create_resilient_llm(settings=s, agent_name="report_node")
        assert isinstance(a, ResilientLLM) and isinstance(b, ResilientLLM)
        assert a._cb is b._cb

    def test_fallback_disabled_returns_primary(self):
        s = _make_settings(fallback_enabled=False)
        result = create_resilient_llm(settings=s)
        assert not isinstance(result, ResilientLLM)


class TestResilientLLMProxy:
    @pytest.mark.asyncio
    async def test_open_circuit_short_circuits_to_fallback(self):
        from unittest.mock import AsyncMock, MagicMock

        primary = MagicMock()
        primary.ainvoke = AsyncMock(side_effect=RuntimeError("primary down"))
        fallback = MagicMock()
        fallback.ainvoke = AsyncMock(return_value="fallback-response")

        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        proxy = ResilientLLM(primary=primary, fallback=fallback, circuit_breaker=breaker)

        # Trip the breaker
        await proxy.ainvoke("ping")
        assert breaker.state == CircuitState.OPEN

        # Next call must go straight to fallback without touching primary
        await proxy.ainvoke("ping")
        assert primary.ainvoke.await_count == 1
        assert fallback.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_success_keeps_circuit_closed(self):
        from unittest.mock import AsyncMock, MagicMock

        primary = MagicMock()
        primary.ainvoke = AsyncMock(return_value="ok")
        fallback = MagicMock()
        fallback.ainvoke = AsyncMock()

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        proxy = ResilientLLM(primary=primary, fallback=fallback, circuit_breaker=breaker)

        await proxy.ainvoke("ping")
        await proxy.ainvoke("ping")
        assert breaker.state == CircuitState.CLOSED
        fallback.ainvoke.assert_not_awaited()
