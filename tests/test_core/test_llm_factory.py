"""Tests for src.core.llm_factory — dual-track channel, resilience wiring, shared breakers."""

from __future__ import annotations

import pytest

from src.core.config import Settings, detect_channel
from src.core.llm_factory import (
    CircuitBreaker,
    CircuitState,
    ResilientLLM,
    _get_shared_breaker,
    _reset_breaker_registry,
    create_llm,
    create_resilient_llm,
)


def _make_settings(**overrides) -> Settings:
    """Build Settings with safe test defaults; no real API calls happen."""
    base = dict(
        llm_model="test-model",
        llm_api_key="sk-test",
        llm_base_url="https://primary.example.com",
        fallback_enabled=True,
        circuit_breaker_threshold=3,
        circuit_breaker_recovery_seconds=60.0,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture(autouse=True)
def _clean_breaker_registry():
    """Isolate the shared breaker registry between tests."""
    _reset_breaker_registry()
    yield
    _reset_breaker_registry()


# ---------------------------------------------------------------------------
# Channel detection
# ---------------------------------------------------------------------------


class TestChannelDetection:
    def test_local_hosts(self):
        assert detect_channel("http://localhost:11434/v1") == "local"
        assert detect_channel("http://127.0.0.1:11434") == "local"
        assert detect_channel("http://[::1]:8000/v1") == "local"
        assert detect_channel("http://0.0.0.0:8080") == "local"

    def test_remote_hosts(self):
        assert detect_channel("https://open.bigmodel.cn/api/paas/v4") == "remote"
        assert detect_channel("https://api.deepseek.com") == "remote"
        assert detect_channel("http://10.0.0.5:8000/v1") == "remote"  # LAN IP ≠ local

    def test_explicit_channel_override(self):
        s = _make_settings(llm_channel="local")
        assert s.channel == "local"
        s = _make_settings(llm_channel="remote")
        assert s.channel == "remote"

    def test_llm_available_local_without_key(self):
        s = _make_settings(llm_base_url="http://localhost:11434", llm_api_key="")
        assert s.llm_available is True

    def test_llm_available_remote_needs_key(self):
        s = _make_settings(llm_api_key="", demo_mode=False)
        assert s.llm_available is False
        # ... unless demo mode tolerates it
        s = _make_settings(llm_api_key="", demo_mode=True)
        assert s.llm_available is True


# ---------------------------------------------------------------------------
# create_llm: track selection and the None sentinel
# ---------------------------------------------------------------------------


class TestCreateLLM:
    def test_returns_none_when_unavailable(self):
        s = _make_settings(llm_api_key="", demo_mode=False)
        assert create_llm(s, role="default") is None

    def test_ollama_native_local(self):
        s = _make_settings(llm_base_url="http://localhost:11434", llm_api_key="")
        llm = create_llm(s, role="default")
        assert type(llm).__name__ == "ChatOllama"
        assert s.ollama_native is True

    def test_vllm_local_openai_compatible(self):
        s = _make_settings(llm_base_url="http://localhost:8000", llm_api_key="")
        assert s.ollama_native is False
        llm = create_llm(s, role="default")
        assert type(llm).__name__ == "ChatOpenAI"
        # /v1 appended for pathless local endpoints
        assert str(llm.openai_api_base).rstrip("/").endswith("/v1")

    def test_compat_mode_forces_openai_protocol(self):
        s = _make_settings(
            llm_base_url="http://localhost:11434", llm_api_key="", llm_compat_mode=True
        )
        assert s.ollama_native is False
        llm = create_llm(s, role="default")
        assert type(llm).__name__ == "ChatOpenAI"

    def test_remote_openai_compatible(self):
        s = _make_settings()
        llm = create_llm(s, role="default")
        assert type(llm).__name__ == "ChatOpenAI"

    def test_role_temperature(self):
        s = _make_settings(role_temperatures={"fast": 0.1, "power": 0.9})
        assert create_llm(s, role="fast").temperature == 0.1
        assert create_llm(s, role="power").temperature == 0.9

    def test_role_model_override(self):
        s = _make_settings(role_models={"power": "bigger-model"})
        assert create_llm(s, role="power").model_name == "bigger-model"
        assert create_llm(s, role="default").model_name == "test-model"

    def test_tier_alias(self):
        """v8 call sites using tier= keep working."""
        s = _make_settings(role_temperatures={"power": 0.9})
        assert create_llm(s, tier="power").temperature == 0.9


# ---------------------------------------------------------------------------
# ResilientLLM wiring
# ---------------------------------------------------------------------------


class TestResilientWiring:
    def test_no_fallback_on_local_channel(self):
        """Local channel degrades deterministically instead of falling back remote."""
        s = _make_settings(
            llm_base_url="http://localhost:11434",
            llm_api_key="",
            fallback_base_url="https://fallback.example.com",
            fallback_model="fallback-model",
        )
        llm = create_resilient_llm(s, role="default")
        assert type(llm).__name__ == "ChatOllama"  # plain, not ResilientLLM

    def test_no_fallback_without_config(self):
        s = _make_settings()
        llm = create_resilient_llm(s, role="default")
        assert type(llm).__name__ == "ChatOpenAI"

    def test_self_fallback_rejected(self):
        s = _make_settings(
            fallback_base_url="https://primary.example.com",
            fallback_model="test-model",  # identical to primary
        )
        llm = create_resilient_llm(s, role="default")
        assert type(llm).__name__ == "ChatOpenAI"  # plain, not ResilientLLM

    def test_cross_vendor_fallback_wraps_resilient(self):
        s = _make_settings(
            fallback_base_url="https://fallback.example.com",
            fallback_model="fallback-model",
            fallback_api_key="sk-fallback",
        )
        llm = create_resilient_llm(s, role="default")
        assert isinstance(llm, ResilientLLM)


# ---------------------------------------------------------------------------
# Circuit breaker mechanics (unchanged from v8, still guarded)
# ---------------------------------------------------------------------------


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
        # Fast-forward monotonic time
        import src.core.llm_factory as lf

        real_monotonic = lf.time.monotonic
        monkeypatch.setattr(lf.time, "monotonic", lambda: real_monotonic() + 100)
        assert cb.state == CircuitState.HALF_OPEN

    def test_shared_breaker_keyed_by_endpoint(self):
        a = _get_shared_breaker("https://x.example.com", "m1", 3, 60.0)
        b = _get_shared_breaker("https://x.example.com", "m1", 3, 60.0)
        c = _get_shared_breaker("https://y.example.com", "m1", 3, 60.0)
        assert a is b
        assert a is not c
