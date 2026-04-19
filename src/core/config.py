"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables with GPA_ prefix."""

    # LLM Configuration
    llm_provider: Literal["openai", "deepseek", "local"] = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_temperature: float = 0.1

    # Fallback model
    fallback_provider: str = "deepseek"
    fallback_model: str = "deepseek-chat"

    # General
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Path("data")
    output_dir: Path = Path("reports")

    model_config = {"env_prefix": "GPA_", "env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
