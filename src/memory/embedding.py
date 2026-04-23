"""Embedding provider for vector memory.

Provides a singleton ``OpenAIEmbeddings`` instance via ``get_embeddings()``.
Falls back to a dummy embedder when ``langchain_openai`` is not installed so
that the application can still start (with degraded search quality).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_embeddings: Any = None
_fallback_mode = False


def get_embeddings() -> Any:
    """Return a singleton embedding client.

    Uses ``langchain_openai.OpenAIEmbeddings`` when available.  Otherwise
    returns ``None`` so callers can detect the fallback and degrade
    gracefully.
    """
    global _embeddings, _fallback_mode
    if _embeddings is not None:
        return _embeddings

    try:
        from langchain_openai import OpenAIEmbeddings  # type: ignore[import-untyped]

        from src.core.config import settings

        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.llm_api_key,
        )
        logger.info("Embeddings initialized: text-embedding-3-small")
    except ImportError:
        logger.warning(
            "langchain_openai not installed – vector search unavailable, "
            "falling back to TF-IDF file-based memory."
        )
        _fallback_mode = True
        _embeddings = None
    except Exception as exc:
        logger.warning("Failed to initialise embeddings: %s – falling back.", exc)
        _fallback_mode = True
        _embeddings = None

    return _embeddings


def is_fallback() -> bool:
    """Return ``True`` if embeddings are unavailable (TF-IDF fallback active)."""
    if _embeddings is None:
        # Trigger lazy init
        get_embeddings()
    return _fallback_mode
