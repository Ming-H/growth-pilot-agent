"""LLM factory for multi-provider support."""

from __future__ import annotations

import enum
import logging
import threading
import time

from langchain_core.language_models import BaseChatModel

from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker that tracks consecutive failures.

    After *failure_threshold* consecutive failures the circuit opens and all
    subsequent calls are short-circuited (routed to fallback).  After
    *recovery_timeout* seconds the circuit transitions to HALF_OPEN, allowing
    one trial call through.  A success resets to CLOSED; a failure reopens.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN


# ---------------------------------------------------------------------------
# Shared breaker registry
# ---------------------------------------------------------------------------

# Circuit breakers are shared per (provider, model, base_url) across the whole
# process.  Without sharing, every call site (plan node, each expert, report
# node) would own a private breaker and a provider-wide outage could never
# trip a common circuit.
_breaker_registry: dict[tuple[str, str, str], CircuitBreaker] = {}
_breaker_registry_lock = threading.Lock()


def _get_shared_breaker(
    provider: str,
    model: str,
    base_url: str,
    failure_threshold: int,
    recovery_timeout: float,
) -> CircuitBreaker:
    """Return the process-wide CircuitBreaker for this LLM endpoint."""
    key = (provider, model, base_url)
    with _breaker_registry_lock:
        breaker = _breaker_registry.get(key)
        if breaker is None:
            breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
            _breaker_registry[key] = breaker
        return breaker


def _reset_breaker_registry() -> None:
    """Clear all shared breakers (for tests)."""
    with _breaker_registry_lock:
        _breaker_registry.clear()


# ---------------------------------------------------------------------------
# Resilient LLM proxy
# ---------------------------------------------------------------------------


class ResilientLLM:
    """Proxy that wraps primary + fallback LLMs with a circuit breaker.

    When the circuit is OPEN all calls go directly to the fallback.  When
    CLOSED or HALF_OPEN the primary is tried first; on failure the call is
    retried against the fallback and the failure is recorded.
    """

    def __init__(
        self,
        primary: BaseChatModel,
        fallback: BaseChatModel,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        # Use object.__setattr__ to avoid triggering __getattr__ proxy logic
        object.__setattr__(self, "_primary", primary)
        object.__setattr__(self, "_fallback", fallback)
        object.__setattr__(self, "_cb", circuit_breaker)

    # -- async interface -----------------------------------------------------

    async def ainvoke(self, *args: object, **kwargs: object) -> object:
        if self._cb.state == CircuitState.OPEN:
            logger.info("Circuit breaker OPEN, using fallback LLM")
            return await self._fallback.ainvoke(*args, **kwargs)
        try:
            result = await self._primary.ainvoke(*args, **kwargs)
            self._cb.record_success()
            return result
        except Exception as exc:
            self._cb.record_failure()
            logger.warning(
                "Primary LLM failed (cb=%s), falling back: %s",
                self._cb.state,
                exc,
            )
            try:
                return await self._fallback.ainvoke(*args, **kwargs)
            except Exception:
                raise exc

    # -- sync interface ------------------------------------------------------

    def invoke(self, *args: object, **kwargs: object) -> object:
        if self._cb.state == CircuitState.OPEN:
            return self._fallback.invoke(*args, **kwargs)
        try:
            result = self._primary.invoke(*args, **kwargs)
            self._cb.record_success()
            return result
        except Exception as exc:
            self._cb.record_failure()
            try:
                return self._fallback.invoke(*args, **kwargs)
            except Exception:
                raise exc

    # -- proxy all other attributes to primary LLM ---------------------------

    def __getattr__(self, name: str) -> object:
        return getattr(self._primary, name)

    @property
    def __class__(self) -> type:  # type: ignore[override]
        return self._primary.__class__


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_llm(
    settings: Settings | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    tier: str | None = None,
    agent_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Create an LLM instance based on configuration.

    Supports: openai, deepseek, local (Ollama).

    ``api_key``/``base_url`` explicitly override the settings-derived values
    (used to build a fallback LLM with its own credentials).
    """
    s = settings or get_settings()

    # If tier is specified, override provider/model/temperature from tier config
    if tier is not None and tier in s.model_tiers:
        tier_config = s.model_tiers[tier]
        p = provider or tier_config.get("provider", s.llm_provider)
        m = model or tier_config.get("model", s.llm_model)
        t = (
            temperature
            if temperature is not None
            else tier_config.get("temperature", s.llm_temperature)
        )
    else:
        p = provider or s.llm_provider
        m = model or s.llm_model
        t = temperature if temperature is not None else s.llm_temperature

    if p in ("openai", "deepseek"):
        from langchain_openai import ChatOpenAI

        resolved_base_url = base_url
        if resolved_base_url is None:
            resolved_base_url = s.llm_base_url or None
            if p == "deepseek" and not resolved_base_url:
                resolved_base_url = "https://api.deepseek.com"

            # If tier config specifies a base_url, prefer it
            if tier is not None and tier in s.model_tiers:
                tier_base_url = s.model_tiers[tier].get("base_url")
                if tier_base_url:
                    resolved_base_url = tier_base_url

        # Fail-fast: only allow placeholder API key in demo mode
        resolved_api_key = api_key or s.llm_api_key
        if not resolved_api_key:
            if s.demo_mode:
                resolved_api_key = "sk-demo-placeholder"
            else:
                raise ValueError("No LLM API key configured. Set GPA_LLM_API_KEY.")
        callbacks = []
        if agent_name:
            from src.core.token_tracker import TokenTrackingCallback

            callbacks = [TokenTrackingCallback(agent_name=agent_name)]

        return ChatOpenAI(
            model=m,
            temperature=t,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            callbacks=callbacks,
        )

    if p == "local":
        from langchain_community.chat_models import ChatOllama

        callbacks = []
        if agent_name:
            from src.core.token_tracker import TokenTrackingCallback

            callbacks = [TokenTrackingCallback(agent_name=agent_name)]

        return ChatOllama(model=m, temperature=t, callbacks=callbacks)

    raise ValueError(f"Unsupported LLM provider: {p}")


def create_resilient_llm(
    settings: Settings | None = None,
    tier: str | None = None,
    agent_name: str | None = None,
) -> BaseChatModel:
    """Create an LLM with fallback and circuit breaker support.

    Returns a :class:`ResilientLLM` proxy when a *genuinely different*
    fallback is configured; otherwise returns the plain primary LLM.

    A fallback identical to the primary (same provider, model and endpoint)
    is detected and rejected — retrying the same failing endpoint adds
    latency without adding resilience.
    """
    s = settings or get_settings()
    primary = create_llm(settings=s, tier=tier, agent_name=agent_name)

    if not s.fallback_enabled:
        return primary

    # Resolve the effective primary endpoint identity for comparison
    tier_config = s.model_tiers.get(tier or "", {})
    primary_provider = tier_config.get("provider", s.llm_provider)
    primary_model = tier_config.get("model", s.llm_model)
    primary_base_url = tier_config.get("base_url", s.llm_base_url)

    # Try to create fallback LLM with its own credentials/endpoint
    try:
        fallback_provider = s.fallback_provider
        fallback_model = s.fallback_model
        if not fallback_provider or not fallback_model:
            return primary
        fallback = create_llm(
            settings=s,
            provider=fallback_provider,
            model=fallback_model,
            api_key=s.fallback_api_key or None,
            base_url=s.fallback_base_url or None,
        )
    except Exception as exc:
        logger.warning("Failed to create fallback LLM, running without fallback: %s", exc)
        return primary

    # Resolve the effective fallback endpoint identity
    fallback_base_url = s.fallback_base_url
    if not fallback_base_url and fallback_provider == "deepseek":
        fallback_base_url = "https://api.deepseek.com"

    # Reject self-fallback: same provider + model + endpoint as the primary
    if (
        fallback_provider == primary_provider
        and fallback_model == primary_model
        and fallback_base_url == primary_base_url
    ):
        logger.warning(
            "Fallback LLM is identical to primary (%s/%s @ %s) — "
            "disabling fallback. Configure a different GPA_FALLBACK_PROVIDER/MODEL.",
            fallback_provider,
            fallback_model,
            fallback_base_url,
        )
        return primary

    # Circuit breakers are shared per endpoint so a provider-wide outage
    # trips one circuit for every call site in the process.
    breaker = _get_shared_breaker(
        provider=primary_provider,
        model=primary_model,
        base_url=primary_base_url or "",
        failure_threshold=s.circuit_breaker_threshold,
        recovery_timeout=s.circuit_breaker_recovery_seconds,
    )

    return ResilientLLM(  # type: ignore[return-value]
        primary=primary,
        fallback=fallback,
        circuit_breaker=breaker,
    )
