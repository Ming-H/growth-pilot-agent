"""LLM factory — dual channel, resolved from a single base_url.

Channel layout (v9):

- **本地通道**（base_url host = localhost/127.0.0.1/::1，零 key）:
  - Ollama-shaped endpoints (:11434 or a path containing /api) use
    ``ChatOllama`` (native protocol — think/tool-calling behave best there).
  - Any other local endpoint (vLLM / SGLang on openEuler, typically :8000)
    is treated as OpenAI-compatible and gets ``/v1`` appended if it has
    no path. ``GPA_LLM_COMPAT_MODE=true`` forces this even for Ollama.
- **远端通道**（其他 host，需 key）: any OpenAI-compatible endpoint —
  Zhipu GLM (``https://open.bigmodel.cn/api/paas/v4`` or
  ``/api/coding/paas/v4``), DeepSeek, OpenAI, … — via ``ChatOpenAI``.

``create_llm`` returns ``None`` when no LLM is configured (remote channel
without a key and not demo mode). That ``None`` is the degradation sentinel
driving every deterministic fallback downstream — absence of a brain is a
mode, not an error.

Roles (fast/default/power) are presets on the ONE configured channel:
they only pick a temperature (Settings.role_temperatures) and optionally a
model override (Settings.role_models) — no per-tier endpoints any more.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from urllib.parse import urlsplit

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

# Circuit breakers are shared per (base_url, model) across the whole process.
# Without sharing, every call site (plan node, each expert, report node)
# would own a private breaker and a provider-wide outage could never trip a
# common circuit.
_breaker_registry: dict[tuple[str, str], CircuitBreaker] = {}
_breaker_registry_lock = threading.Lock()


def _get_shared_breaker(
    base_url: str,
    model: str,
    failure_threshold: int,
    recovery_timeout: float,
) -> CircuitBreaker:
    """Return the process-wide CircuitBreaker for this LLM endpoint."""
    key = (base_url, model)
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
# Resilient LLM proxy (remote → remote fallback only)
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
# Factory
# ---------------------------------------------------------------------------


def _openai_compatible_local_url(base_url: str) -> str:
    """Normalise a local OpenAI-compatible base_url (append /v1 if pathless)."""
    parts = urlsplit(base_url)
    if not parts.path or parts.path == "/":
        return base_url.rstrip("/") + "/v1"
    return base_url


def create_llm(
    settings: Settings | None = None,
    *,
    role: str = "default",
    agent_name: str | None = None,
    tier: str | None = None,  # legacy alias for `role` (v8 call sites)
) -> BaseChatModel | None:
    """Create the LLM for the configured channel and role preset.

    Returns ``None`` when no LLM is available (remote channel, no key, not
    demo mode) — callers treat ``None`` as the deterministic-degradation
    sentinel rather than an error.
    """
    s = settings or get_settings()
    if tier is not None:
        role = tier  # v8 compatibility

    if not s.llm_available:
        logger.info(
            "[llm_factory] No LLM configured (channel=%s, key=%s, demo=%s) — "
            "deterministic degradation",
            s.channel,
            bool(s.llm_api_key),
            s.demo_mode,
        )
        return None

    model = s.role_model(role)
    temperature = s.role_temperature(role)

    callbacks = []
    if agent_name:
        from src.core.token_tracker import TokenTrackingCallback

        callbacks = [TokenTrackingCallback(agent_name=agent_name)]

    if s.ollama_native:
        from langchain_ollama import ChatOllama

        # ChatOllama expects the host root (no /v1, no /api suffix).
        parts = urlsplit(s.llm_base_url)
        host = f"{parts.scheme}://{parts.netloc}"
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=host,
            callbacks=callbacks,  # type: ignore[arg-type]
        )

    # Everything else (local vLLM/SGLang + all remote endpoints) is
    # OpenAI-compatible.
    from langchain_openai import ChatOpenAI

    base_url = s.llm_base_url
    if s.channel == "local":
        base_url = _openai_compatible_local_url(base_url)

    api_key = s.llm_api_key
    if s.channel == "local" and not api_key:
        api_key = "local-not-needed"

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        callbacks=callbacks,
    )


def create_resilient_llm(
    settings: Settings | None = None,
    *,
    role: str = "default",
    agent_name: str | None = None,
    tier: str | None = None,  # legacy alias for `role`
) -> BaseChatModel | None:
    """Create the role LLM, wrapped with cross-vendor fallback when useful.

    Fallback only matters remote→remote: on the local channel the honest
    "fallback" is deterministic degradation (``None`` handling downstream),
    and a remote fallback would silently reintroduce network dependency the
    local track exists to avoid. A fallback identical to the primary (same
    base_url + model) is detected and rejected.
    """
    s = settings or get_settings()
    if tier is not None:
        role = tier

    primary = create_llm(s, role=role, agent_name=agent_name)
    if primary is None:
        return None

    if not s.fallback_enabled or s.channel != "remote":
        return primary
    if not s.fallback_base_url or not s.fallback_model:
        return primary

    if s.fallback_base_url.rstrip("/") == s.llm_base_url.rstrip(
        "/"
    ) and s.fallback_model == s.role_model(role):
        logger.warning(
            "Fallback LLM is identical to primary (%s/%s) — disabling fallback. "
            "Configure a different GPA_FALLBACK_BASE_URL/MODEL.",
            s.fallback_base_url,
            s.fallback_model,
        )
        return primary

    try:
        fallback = create_fallback_llm(
            s,
            base_url=s.fallback_base_url,
            model=s.fallback_model,
            api_key=s.fallback_api_key or None,
            agent_name=agent_name,
        )
    except Exception as exc:
        logger.warning("Failed to create fallback LLM, running without fallback: %s", exc)
        return primary

    breaker = _get_shared_breaker(
        base_url=s.llm_base_url,
        model=s.role_model(role),
        failure_threshold=s.circuit_breaker_threshold,
        recovery_timeout=s.circuit_breaker_recovery_seconds,
    )
    return ResilientLLM(  # type: ignore[return-value]
        primary=primary,
        fallback=fallback,
        circuit_breaker=breaker,
    )


def create_fallback_llm(
    s: Settings,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    agent_name: str | None,
) -> BaseChatModel:
    """Build a plain ChatOpenAI against an explicit endpoint (fallback path)."""
    from langchain_openai import ChatOpenAI

    callbacks = []
    if agent_name:
        from src.core.token_tracker import TokenTrackingCallback

        callbacks = [TokenTrackingCallback(agent_name=agent_name)]

    effective_key = api_key or s.llm_api_key
    if not effective_key:
        if s.demo_mode:
            effective_key = "sk-demo-placeholder"
        else:
            raise ValueError("Fallback endpoint needs an API key (GPA_FALLBACK_API_KEY).")

    return ChatOpenAI(
        model=model,
        temperature=s.role_temperature("default"),
        api_key=effective_key,
        base_url=base_url,
        callbacks=callbacks,
    )
