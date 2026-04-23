"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables with GPA_ prefix."""

    # LLM Configuration
    llm_provider: Literal["openai", "deepseek", "local"] = "openai"
    llm_model: str = "glm-4.7"
    llm_api_key: str = ""
    llm_base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4"
    llm_temperature: float = 0.1

    # Fallback model
    fallback_provider: str = "openai"
    fallback_model: str = "glm-4.7"

    # Model tiers configuration
    model_tiers: dict[str, dict[str, Any]] = {
        "fast": {"provider": "openai", "model": "glm-4.7", "temperature": 0.3, "base_url": "https://open.bigmodel.cn/api/coding/paas/v4"},
        "default": {"provider": "openai", "model": "glm-4.7", "temperature": 0.5, "base_url": "https://open.bigmodel.cn/api/coding/paas/v4"},
        "power": {"provider": "openai", "model": "glm-4.7", "temperature": 0.7, "base_url": "https://open.bigmodel.cn/api/coding/paas/v4"},
    }

    # Memory system
    memory_base_path: str = "./data/memory"

    # Retry configuration
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    # General
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Path("data")
    output_dir: Path = Path("reports")

    # Chief Agent configuration
    chief_max_iterations: int = 10
    chief_model_tier: str = "power"
    expert_model_tier: str = "default"
    chief_enable_evaluator: bool = False
    chief_max_plan_steps: int = 6

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

    # Observability
    otel_enabled: bool = False
    otel_service_name: str = "growth-pilot-agent"

    model_config = {"env_prefix": "GPA_", "env_file": ".env", "extra": "ignore"}


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses functools.lru_cache so that the Settings object (which reads env
    vars / .env at construction time) is created only once per process.
    Call ``get_settings.cache_clear()`` to force a reload (e.g. in tests).
    """
    return Settings()
