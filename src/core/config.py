"""Configuration management using Pydantic Settings.

v9 LLM 配置（tally 式双轨通道，换脑不改代码）:

- ``GPA_LLM_BASE_URL`` 是通道开关，按 host 自动识别:
  * localhost / 127.0.0.1 / ::1 → **本地通道**（零 key 零费用零外呼）
      - Ollama 形态（:11434 或路径含 /api）→ Ollama 原生协议
      - 其余（vLLM / SGLang 的 OpenAI 兼容 /v1）→ OpenAI 兼容
  * 其他 host → **远端通道**（智谱 GLM / DeepSeek / 任意 OpenAI 兼容端点，需 key）
- 角色预设（fast/default/power）只影响温度与可选的模型覆盖，不再整端点配置。
- 显式覆盖: ``GPA_LLM_CHANNEL=local|remote``；``GPA_LLM_COMPAT_MODE=true``
  强制对 Ollama 也走 OpenAI 兼容 /v1（逃生门）。
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import model_validator
from pydantic_settings import BaseSettings

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"})


def detect_channel(base_url: str) -> Literal["local", "remote"]:
    """Classify a base_url as a local or remote channel by its host."""
    try:
        host = urlsplit(base_url).hostname or ""
    except ValueError:
        return "remote"
    return "local" if host.lower() in _LOCAL_HOSTS else "remote"


def _memory_dir() -> str:
    """Directory containing the memory store (parent of memory_db_path).

    Bridge for the v8 MemoryManager(base_path=...) call sites until the v9
    SQLite MemoryStore fully replaces it.
    """
    from pathlib import Path

    return str(Path(get_settings().memory_db_path).parent)


def _is_ollama_native(base_url: str) -> bool:
    """True when the local endpoint speaks Ollama's native protocol.

    Heuristic (tally-proven): default port 11434 or a path containing
    ``/api`` (Ollama native mounts under /api, not /v1).
    """
    try:
        parts = urlsplit(base_url)
    except ValueError:
        return False
    if parts.port == 11434:
        return True
    return "/api" in (parts.path or "")


class Settings(BaseSettings):
    """Application settings, loaded from environment variables with GPA_ prefix."""

    # ── LLM 双轨通道 ─────────────────────────────────────────────────────
    # THE channel switch. Default: Zhipu GLM (OpenAI-compatible, needs key).
    # Point it at http://localhost:11434 (Ollama) or http://localhost:8000
    # (vLLM) for the zero-key local track.
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model: str = "glm-4.7"
    llm_api_key: str = ""  # empty = OK for localhost; required for remote
    llm_channel: Literal["auto", "local", "remote"] = "auto"
    # Force OpenAI-compatible /v1 even against Ollama (escape hatch for
    # setups where the native protocol misbehaves).
    llm_compat_mode: bool = False
    llm_temperature: float = 0.3
    # Role presets: temperature per role; optional per-role model override
    # (empty model → the single channel model).
    role_temperatures: dict[str, float] = {"fast": 0.2, "default": 0.5, "power": 0.7}
    role_models: dict[str, str] = {}

    # ── Fallback（仅远端→远端有意义；本地通道的"兜底"是确定性降级）────────
    fallback_enabled: bool = True
    fallback_base_url: str = ""
    fallback_model: str = ""
    fallback_api_key: str = ""

    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery_seconds: float = 60.0

    # ── Memory（v9 三层：SQLite，路径见 memory_db_path）──────────────────
    memory_db_path: str = "./data/memory/memory.db"

    # Retry configuration
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    # General
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Path("data")
    output_dir: Path = Path("reports")

    # ── Orchestration ───────────────────────────────────────────────────
    #   "agent"    — Chief Agent tool-calling loop (LLM decides tools/ordering)
    #   "workflow" — fixed plan→execute→evaluate DAG (deterministic routing)
    orchestration_mode: Literal["agent", "workflow"] = "agent"

    # Graph configuration
    max_refinement_rounds: int = 2
    quality_threshold: float = 0.7
    budget_approval_threshold: float = 10_000
    # Must cover the deterministic pipeline AND the LLM synthesis call —
    # a single slow LLM call can take 60-90s, so keep this well above it.
    expert_timeout_seconds: int = 300
    # Hard cap on agent⇄tools round trips (weak local models can loop).
    agent_max_steps: int = 12

    # ── 对账闸门（KPI reconciliation gate）───────────────────────────────
    #   on       — violations replace the report with the deterministic
    #               template + reconciliation appendix (strict, default)
    #   annotate — keep the report, append a discrepancy table
    #   off      — skip the gate
    kpi_gate: Literal["on", "annotate", "off"] = "on"
    # Tolerated fraction of untraceable KPI claims before the gate trips
    # (0.0 = any single violation trips; 1.0 = never trips).
    kpi_gate_threshold: float = 0.0

    # SSE streaming
    sse_timeout_seconds: int = 300

    # Database
    db_url: str = "postgresql+asyncpg://gpa:gpa@localhost:5432/growth_pilot"
    db_test_url: str = "sqlite+aiosqlite:///./test.db"
    db_echo: bool = False
    db_pool_size: int = 10

    # JWT Auth
    jwt_secret: str = "change-me-in-production"
    jwt_expire_hours: int = 24
    jwt_algorithm: str = "HS256"

    # Demo mode
    demo_mode: bool = False

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Observability
    otel_enabled: bool = False
    otel_service_name: str = "growth-pilot-agent"

    # LangSmith tracing
    langsmith_enabled: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "growth-pilot"

    model_config = {
        "env_prefix": "GPA_",
        "env_file": ".env",
        "extra": "ignore",
        "protected_namespaces": ("settings_",),
    }

    # ------------------------------------------------------------------
    # Channel resolution
    # ------------------------------------------------------------------

    @property
    def channel(self) -> Literal["local", "remote"]:
        """Resolved channel: explicit llm_channel, else auto-detect from host."""
        if self.llm_channel != "auto":
            return self.llm_channel  # type: ignore[return-value]
        return detect_channel(self.llm_base_url)

    @property
    def llm_available(self) -> bool:
        """Whether an LLM brain is configured.

        Local endpoints are key-free (always available config-wise; runtime
        failures degrade point-wise). Remote endpoints need a key — demo
        mode does NOT fake availability: a keyless remote endpoint would
        only fail at call time (the v8 dishonesty this replaces).
        """
        if self.channel == "local":
            return True
        return bool(self.llm_api_key)

    @property
    def ollama_native(self) -> bool:
        """Whether the local channel should speak Ollama's native protocol."""
        return (
            self.channel == "local"
            and not self.llm_compat_mode
            and _is_ollama_native(self.llm_base_url)
        )

    def role_temperature(self, role: str) -> float:
        return self.role_temperatures.get(role, self.llm_temperature)

    def role_model(self, role: str) -> str:
        return self.role_models.get(role) or self.llm_model

    @model_validator(mode="after")
    def _validate_security(self) -> "Settings":
        if not self.demo_mode and self.jwt_secret == "change-me-in-production":
            import logging

            logging.getLogger(__name__).warning(
                "Default JWT secret detected in non-demo mode. "
                "Set GPA_JWT_SECRET to a secure random string."
            )
        return self


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses functools.lru_cache so that the Settings object (which reads env
    vars / .env at construction time) is created only once per process.
    Call ``get_settings.cache_clear()`` to force a reload (e.g. in tests).
    """
    return Settings()
