"""Hook system for agent lifecycle management."""
from __future__ import annotations

import abc
import logging
import time

logger = logging.getLogger(__name__)


class PreRunHook(abc.ABC):
    """Hook executed before agent run."""

    @abc.abstractmethod
    async def on_pre_run(self, agent_name: str, state: dict) -> dict:
        ...


class PostRunHook(abc.ABC):
    """Hook executed after agent run."""

    @abc.abstractmethod
    async def on_post_run(self, agent_name: str, result: dict, state: dict) -> dict:
        ...


class TracingHook(PostRunHook):
    """Record agent execution timing."""

    async def on_pre_run(self, agent_name, state):
        state["_trace_start"] = time.time()
        return state

    async def on_post_run(self, agent_name, result, state):
        start = state.pop("_trace_start", time.time())
        result["_trace"] = {"agent": agent_name, "elapsed_s": round(time.time() - start, 2)}
        return result


class LoggingHook(PreRunHook, PostRunHook):
    """Unified logging."""

    async def on_pre_run(self, agent_name, state):
        logger.info("[%s] Starting execution", agent_name)
        return state

    async def on_post_run(self, agent_name, result, state):
        status = "with errors" if result.get("errors") else "successfully"
        logger.info("[%s] Completed %s. Keys: %s", agent_name, status, list(result.keys()))
        return result


class MetricsHook(PostRunHook):
    """Collect execution metrics."""

    async def on_post_run(self, agent_name, result, state):
        from datetime import datetime

        result["_metrics"] = {
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "has_errors": bool(result.get("errors")),
        }
        return result
