"""Compiled-graph runtime cache (A6 fix).

Both orchestration modes compiled once per process, checkpointer
initialised once — run_agent_analysis used to rebuild the graph and
re-initialise the Postgres saver on every run.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_compiled: dict[str, Any] = {}
_lock = asyncio.Lock()


async def get_compiled_graph(mode: str = "agent") -> Any:
    """Return the cached compiled graph for *mode* ("agent" | "workflow")."""
    async with _lock:
        if mode not in _compiled:
            if mode == "agent":
                from src.agent.graph import build_compiled_agent_graph

                _compiled[mode] = await build_compiled_agent_graph()
            else:
                from src.graph.graph import build_compiled_graph

                _compiled[mode] = await build_compiled_graph()
            logger.info("[runtime] compiled graph cached (mode=%s)", mode)
        return _compiled[mode]


def clear_graph_cache() -> None:
    """Drop cached graphs (tests)."""
    _compiled.clear()
