"""Tests for src.core.config - Settings, channel resolution, get_settings."""

from __future__ import annotations

import os

from src.core.config import Settings, get_settings


class TestSettings:
    """Tests for the Settings model."""

    def test_settings_creates_instance(self):
        """Settings can be instantiated with defaults."""
        s = Settings(_env_file=None)
        assert isinstance(s, Settings)
        assert s.llm_model == "glm-4.7"
        assert isinstance(s.llm_temperature, float)
        assert s.log_level == "INFO"
        assert s.max_retries == 3
        assert s.max_refinement_rounds == 2
        assert s.quality_threshold == 0.7
        assert s.budget_approval_threshold == 10_000

    def test_settings_custom_values(self):
        """Settings accepts custom values."""
        s = Settings(
            _env_file=None,
            llm_base_url="https://api.deepseek.com",
            llm_model="deepseek-chat",
            llm_temperature=0.5,
        )
        assert s.llm_model == "deepseek-chat"
        assert s.llm_temperature == 0.5
        assert s.channel == "remote"

    def test_default_channel_is_remote_bigmodel(self):
        """The default config points at Zhipu GLM (remote track)."""
        s = Settings(_env_file=None)
        assert s.channel == "remote"
        assert "bigmodel.cn" in s.llm_base_url

    def test_role_presets_default(self):
        """Role temperature presets cover fast/default/power."""
        s = Settings(_env_file=None)
        for role in ("fast", "default", "power"):
            assert role in s.role_temperatures
        assert s.role_models == {}

    def test_settings_env_prefix(self):
        """Settings reads from GPA_* environment variables."""
        os.environ["GPA_LLM_MODEL"] = "gpt-3.5-turbo"
        try:
            s = Settings(_env_file=None)
            assert s.llm_model == "gpt-3.5-turbo"
        finally:
            del os.environ["GPA_LLM_MODEL"]

    def test_settings_env_channel_switch(self):
        """GPA_LLM_BASE_URL alone flips the channel (换脑不改代码)."""
        os.environ["GPA_LLM_BASE_URL"] = "http://localhost:11434"
        try:
            s = Settings(_env_file=None, llm_api_key="")
            assert s.channel == "local"
            assert s.llm_available is True
        finally:
            del os.environ["GPA_LLM_BASE_URL"]

    def test_settings_path_fields(self):
        """data_dir and output_dir default to Path objects."""
        s = Settings(_env_file=None)
        assert isinstance(s.data_dir, type(s.data_dir))
        assert isinstance(s.output_dir, type(s.output_dir))

    def test_fallback_defaults_off(self):
        """Cross-vendor fallback is opt-in via explicit endpoint config."""
        s = Settings(_env_file=None)
        assert s.fallback_base_url == ""
        assert s.fallback_model == ""

    def test_kpi_gate_defaults(self):
        s = Settings(_env_file=None)
        assert s.kpi_gate == "on"
        assert s.kpi_gate_threshold == 0.0

    def test_memory_db_path_default(self):
        s = Settings(_env_file=None)
        assert s.memory_db_path.endswith("memory.db")

    def test_retry_config_defaults(self):
        s = Settings(_env_file=None)
        assert s.retry_min_wait == 1.0
        assert s.retry_max_wait == 10.0

    def test_agent_max_steps_default(self):
        s = Settings(_env_file=None)
        assert s.agent_max_steps == 12


class TestGetSettings:
    """Tests for the get_settings cached factory."""

    def test_get_settings_returns_settings(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_get_settings_cache_clear(self):
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        assert s1 is not s2
