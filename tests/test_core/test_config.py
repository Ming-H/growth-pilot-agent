"""Tests for src.core.config - Settings and get_settings."""

from __future__ import annotations

import os

import pytest

from src.core.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings model."""

    def test_settings_creates_instance(self):
        """Settings can be instantiated with defaults."""
        s = Settings()
        assert isinstance(s, Settings)
        assert s.llm_provider == "openai"
        assert s.llm_model == "glm-4.7"
        assert isinstance(s.llm_temperature, float)
        assert s.log_level == "INFO"
        assert s.max_retries == 3
        # New graph config fields
        assert s.max_refinement_rounds == 2
        assert s.quality_threshold == 0.7
        assert s.budget_approval_threshold == 10_000

    def test_settings_custom_values(self):
        """Settings accepts custom values."""
        s = Settings(
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            llm_temperature=0.5,
        )
        assert s.llm_provider == "deepseek"
        assert s.llm_model == "deepseek-chat"
        assert s.llm_temperature == 0.5

    def test_model_tiers_default(self):
        """Default model_tiers contain fast/default/power tiers."""
        s = Settings()
        assert "fast" in s.model_tiers
        assert "default" in s.model_tiers
        assert "power" in s.model_tiers

        # Verify structure of each tier
        for tier_name in ("fast", "default", "power"):
            tier = s.model_tiers[tier_name]
            assert "provider" in tier
            assert "model" in tier
            assert "temperature" in tier

    def test_model_tiers_fast_uses_configured_provider(self):
        """The 'fast' tier uses the configured provider."""
        s = Settings()
        assert s.model_tiers["fast"]["provider"] in ("openai", "deepseek")

    def test_model_tiers_power_uses_configured_model(self):
        """The 'power' tier uses the configured model."""
        s = Settings()
        assert s.model_tiers["power"]["model"] in ("gpt-4o", "glm-4.7")

    def test_settings_env_prefix(self):
        """Settings reads from GPA_* environment variables."""
        os.environ["GPA_LLM_MODEL"] = "gpt-3.5-turbo"
        try:
            s = Settings()
            assert s.llm_model == "gpt-3.5-turbo"
        finally:
            del os.environ["GPA_LLM_MODEL"]

    def test_settings_path_fields(self):
        """data_dir and output_dir default to Path objects."""
        s = Settings()
        assert isinstance(s.data_dir, type(s.data_dir))
        assert isinstance(s.output_dir, type(s.output_dir))

    def test_fallback_defaults(self):
        """Fallback provider/model have sensible defaults."""
        s = Settings()
        assert s.fallback_provider in ("deepseek", "openai")
        assert isinstance(s.fallback_model, str) and len(s.fallback_model) > 0

    def test_memory_base_path_default(self):
        """Memory base path has a default."""
        s = Settings()
        assert s.memory_base_path == "./data/memory"

    def test_retry_config_defaults(self):
        """Retry configuration has sensible defaults."""
        s = Settings()
        assert s.retry_min_wait == 1.0
        assert s.retry_max_wait == 10.0


class TestGetSettings:
    """Tests for the get_settings cached factory."""

    def test_get_settings_returns_settings(self):
        """get_settings returns a Settings instance."""
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_cached(self):
        """Repeated calls return the same object (lru_cache)."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_get_settings_cache_clear(self):
        """After cache_clear, a new instance is created."""
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        assert s1 is not s2
