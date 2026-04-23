"""LLM factory for multi-provider support."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def create_llm(
    settings: Settings | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    tier: str | None = None,
) -> BaseChatModel:
    """Create an LLM instance based on configuration.

    Supports: openai, deepseek, local (Ollama).
    """
    s = settings or get_settings()

    # If tier is specified, override provider/model/temperature from tier config
    if tier is not None and tier in s.model_tiers:
        tier_config = s.model_tiers[tier]
        p = provider or tier_config.get("provider", s.llm_provider)
        m = model or tier_config.get("model", s.llm_model)
        t = temperature if temperature is not None else tier_config.get("temperature", s.llm_temperature)
    else:
        p = provider or s.llm_provider
        m = model or s.llm_model
        t = temperature if temperature is not None else s.llm_temperature

    if p in ("openai", "deepseek"):
        from langchain_openai import ChatOpenAI

        base_url = s.llm_base_url or None
        if p == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com"

        # If tier config specifies a base_url, prefer it
        if tier is not None and tier in s.model_tiers:
            tier_base_url = s.model_tiers[tier].get("base_url")
            if tier_base_url:
                base_url = tier_base_url

        # Allow creation without API key (demo mode);
        # actual LLM calls will fail gracefully in Agent._invoke_llm
        api_key = s.llm_api_key or "sk-demo-placeholder"
        return ChatOpenAI(
            model=m,
            temperature=t,
            api_key=api_key,
            base_url=base_url,
        )

    if p == "local":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model=m, temperature=t)

    raise ValueError(f"Unsupported LLM provider: {p}")



