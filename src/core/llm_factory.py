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
) -> BaseChatModel:
    """Create an LLM instance based on configuration.

    Supports: openai, deepseek, local (Ollama).
    """
    s = settings or get_settings()
    p = provider or s.llm_provider
    m = model or s.llm_model
    t = temperature if temperature is not None else s.llm_temperature

    if p in ("openai", "deepseek"):
        from langchain_openai import ChatOpenAI

        base_url = s.llm_base_url or None
        if p == "deepseek" and not base_url:
            base_url = "https://api.deepseek.com"

        return ChatOpenAI(
            model=m,
            temperature=t,
            api_key=s.llm_api_key or None,
            base_url=base_url,
        )

    if p == "local":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model=m, temperature=t)

    raise ValueError(f"Unsupported LLM provider: {p}")


def create_llm_with_fallback(settings: Settings | None = None) -> BaseChatModel:
    """Create LLM with automatic fallback on failure.

    Uses LangChain's fallback mechanism:
    primary model → fallback model.
    """
    s = settings or get_settings()
    primary = create_llm(s)
    fallback = create_llm(
        s,
        provider=s.fallback_provider,
        model=s.fallback_model,
    )
    return primary.with_fallbacks([fallback])
