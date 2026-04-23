"""Checkpoint management for LangGraph StateGraph.

Provides a factory that returns an appropriate checkpointer based on
configuration:

- Production mode (demo_mode=False): uses ``AsyncPostgresSaver`` backed by
  the PostgreSQL database configured in ``settings.db_url``.
- Demo / testing mode (demo_mode=True): uses ``MemorySaver`` (in-process,
  no external dependencies).

If the ``langgraph-checkpoint-postgres`` package is not installed, the factory
gracefully falls back to ``MemorySaver`` regardless of the demo_mode flag.
"""
from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# Try to import the async PostgresSaver.  The package is optional; if missing
# we fall back to MemorySaver.
_PostgresSaver_available = False
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _PostgresSaver_available = True
except ImportError:
    AsyncPostgresSaver = None  # type: ignore[assignment,misc]


async def get_checkpointer():  # noqa: ANN202 – return type depends on backend
    """Create and return an initialised checkpointer.

    Returns:
        An ``AsyncPostgresSaver`` (production) or ``MemorySaver`` (demo/fallback)
        ready to be passed as ``checkpointer=`` to ``StateGraph.compile()``.
    """
    from src.core.config import get_settings

    settings = get_settings()

    # --- Demo mode or missing PostgresSaver -> MemorySaver ----------------
    if settings.demo_mode or not _PostgresSaver_available:
        reason = "demo_mode=True" if settings.demo_mode else "langgraph-checkpoint-postgres not installed"
        logger.info("[checkpoint] Using MemorySaver (%s)", reason)
        return MemorySaver()

    # --- Production mode: AsyncPostgresSaver ------------------------------
    # The settings.db_url uses SQLAlchemy-style URLs
    # (e.g. ``postgresql+asyncpg://user:pass@host/db``).  AsyncPostgresSaver
    # expects a plain connection string compatible with ``psycopg`` /
    # ``asyncpg``, so strip the ``+asyncpg`` driver portion.
    conn_string = settings.db_url.replace("+asyncpg", "")

    try:
        checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)
        await checkpointer.setup()
        logger.info("[checkpoint] AsyncPostgresSaver initialised (db_url=%s)", conn_string.split("@")[-1])
        return checkpointer
    except Exception as exc:
        logger.warning(
            "[checkpoint] Failed to initialise AsyncPostgresSaver (%s). "
            "Falling back to MemorySaver.",
            exc,
        )
        return MemorySaver()
