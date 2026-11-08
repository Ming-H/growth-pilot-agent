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

# Keeps the async context manager of a live Postgres saver alive for the
# process lifetime (see get_checkpointer).
_pg_saver_cm = None


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
        reason = (
            "demo_mode=True"
            if settings.demo_mode
            else "langgraph-checkpoint-postgres not installed"
        )
        logger.info("[checkpoint] Using MemorySaver (%s)", reason)
        return MemorySaver()

    # --- SQLite backend -> MemorySaver (A7: no doomed Postgres attempt) ---
    if settings.db_url.startswith("sqlite"):
        logger.info("[checkpoint] SQLite db_url -> MemorySaver (Postgres-only saver)")
        return MemorySaver()

    # --- Production mode: AsyncPostgresSaver ------------------------------
    # The settings.db_url uses SQLAlchemy-style URLs
    # (e.g. ``postgresql+asyncpg://user:pass@host/db``).  AsyncPostgresSaver
    # expects a plain connection string compatible with ``psycopg`` /
    # ``asyncpg`` — strip any SQLAlchemy driver suffix (A7: asyncpg-only
    # replace left ``+psycopg://`` URLs invalid).
    import re as _re

    conn_string = _re.sub(r"\+(asyncpg|psycopg2|psycopg)", "", settings.db_url)

    try:
        # ``from_conn_string`` returns an async context manager. The saver
        # must outlive this function (process-level lifetime), so enter the
        # context manually and keep a module-level reference to prevent
        # premature cleanup by the garbage collector.
        cm = AsyncPostgresSaver.from_conn_string(conn_string)
        try:
            checkpointer = await cm.__aenter__()
            await checkpointer.setup()
        except Exception:
            # A7: __aenter__ succeeded but setup() failed — exit the CM we
            # opened, or the connection leaks while we fall back.
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass
            raise
        global _pg_saver_cm
        _pg_saver_cm = cm
        logger.info(
            "[checkpoint] AsyncPostgresSaver initialised (db_url=%s)", conn_string.split("@")[-1]
        )
        return checkpointer
    except Exception as exc:
        logger.warning(
            "[checkpoint] Failed to initialise AsyncPostgresSaver (%s). "
            "Falling back to MemorySaver.",
            exc,
        )
        return MemorySaver()
