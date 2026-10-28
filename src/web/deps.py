"""Shared web-layer dependencies: memory store + compiled graphs."""

from __future__ import annotations

import logging
import threading
from typing import Any

from src.core.config import get_settings
from src.memory import MemoryStore

logger = logging.getLogger(__name__)

# Single process-wide MemoryStore (SQLite handles per-org rows natively).
_memory_store: MemoryStore | None = None
_memory_lock = threading.Lock()


def get_memory() -> MemoryStore:
    """Return the process-wide MemoryStore instance (lazy, cached)."""
    global _memory_store
    if _memory_store is None:
        with _memory_lock:
            if _memory_store is None:
                _memory_store = MemoryStore(db_path=get_settings().memory_db_path)
    return _memory_store


async def get_graph_for(mode: str | None = None) -> Any:
    """Return the cached compiled graph for the resolved mode.

    A2 fix: SSE and the sync path both dispatch on the SAME mode
    resolution (Settings.orchestration_mode unless overridden).
    get_compiled_graph owns the process-wide cache and lock.
    """
    from src.core.config import get_settings as _gs
    from src.graph.runtime import get_compiled_graph

    resolved = mode or _gs().orchestration_mode
    return await get_compiled_graph(resolved)


def thread_id_for(org_id: str, analysis_id: str) -> str:
    """Canonical HITL thread id — every path (run/state/approve/resume)
    MUST use this (A3: the four endpoints used to disagree, so approval
    could never find the paused checkpoint)."""
    return f"{org_id}:{analysis_id}"
