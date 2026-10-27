"""Run-scoped token usage tracking via LangChain callback handler.

v9: attribution is **per run**, not per process. The tracker lives in a
ContextVar set by ``cost_scope()``; ``asyncio.to_thread`` propagates context
vars into worker threads, so expert pipelines executing there attribute to
the right run. Records made outside any scope are dropped (debug-logged) —
never attributed to the wrong run.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class CostTracker:
    """Track token usage and estimated cost per agent for one run."""

    PRICING: dict[str, dict[str, float]] = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "glm-4.7": {"input": 0.000001, "output": 0.000002},
    }

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record(self, agent: str, model: str, input_tokens: int, output_tokens: int) -> None:
        prices = self.PRICING.get(model, {"input": 0.50, "output": 1.50})
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
        self._records.append(
            {
                "agent": agent,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
            }
        )

    @property
    def total_cost(self) -> float:
        return round(float(sum(r["cost_usd"] for r in self._records)), 6)

    @property
    def total_tokens(self) -> int:
        return sum(r["input_tokens"] + r["output_tokens"] for r in self._records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r["input_tokens"] for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r["output_tokens"] for r in self._records)

    def report(self) -> dict[str, Any]:
        from collections import defaultdict

        by_agent: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"cost_usd": 0.0, "calls": 0, "tokens": 0}
        )
        for r in self._records:
            by_agent[r["agent"]]["cost_usd"] += r["cost_usd"]
            by_agent[r["agent"]]["calls"] += 1
            by_agent[r["agent"]]["tokens"] += r["input_tokens"] + r["output_tokens"]
        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_calls": len(self._records),
            "by_agent": dict(by_agent),
        }

    def reset(self) -> None:
        self._records.clear()


# ---------------------------------------------------------------------------
# Run-scoped tracker (ContextVar)
# ---------------------------------------------------------------------------

run_cost_tracker: ContextVar[CostTracker | None] = ContextVar("run_cost_tracker", default=None)


@asynccontextmanager
async def cost_scope() -> AsyncIterator[CostTracker]:
    """Open a per-run cost attribution scope.

    Every LLM call made inside this context (including from worker threads —
    ``asyncio.to_thread`` propagates context vars) records to this tracker.
    """
    tracker = CostTracker()
    token = run_cost_tracker.set(tracker)
    try:
        yield tracker
    finally:
        run_cost_tracker.reset(token)


def get_cost_tracker() -> CostTracker:
    """Return the current run's tracker, or a fresh detached one.

    The detached fallback keeps legacy callers (tests, report nodes outside
    a scope) working; a fresh tracker simply reports zeros rather than
    mis-attributing another run's records.
    """
    return run_cost_tracker.get() or CostTracker()


def reset_cost_tracker() -> None:
    """No-op retained for call-site compatibility.

    v8 reset a process-global singleton at run start, which corrupted
    attribution under concurrent requests. v9 scopes trackers per run via
    :func:`cost_scope` — nothing to reset.
    """
    return None


class TokenTrackingCallback(BaseCallbackHandler):
    """LangChain callback handler that records token usage to the run tracker."""

    def __init__(self, agent_name: str = "unknown") -> None:
        self._agent_name = agent_name

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        tracker = run_cost_tracker.get()
        if tracker is None:
            # Outside any run scope (e.g. a long-lived cached expert LLM used
            # ad hoc) — drop rather than mis-attribute.
            logger.debug("[token_tracker] LLM call outside cost_scope, not recorded")
            return
        try:
            # Extract token usage from response metadata
            for generation_list in response.generations:
                for generation in generation_list:
                    meta = getattr(generation, "message", None)
                    if meta:
                        response_metadata = getattr(meta, "response_metadata", {}) or {}
                        token_usage = response_metadata.get("token_usage", {})
                        if not token_usage:
                            token_usage = response_metadata.get("usage", {})
                        if token_usage:
                            input_tokens = token_usage.get("prompt_tokens", 0)
                            output_tokens = token_usage.get("completion_tokens", 0)
                            if input_tokens or output_tokens:
                                fallback_model = (
                                    response.llm_output.get("model_name", "unknown")
                                    if response.llm_output
                                    else "unknown"
                                )
                                model = response_metadata.get("model_name", fallback_model)
                                tracker.record(self._agent_name, model, input_tokens, output_tokens)
                                return

            # Try llm_output as fallback
            if response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                model = response.llm_output.get("model_name", "unknown")
                if token_usage:
                    input_tokens = token_usage.get("prompt_tokens", 0)
                    output_tokens = token_usage.get("completion_tokens", 0)
                    if input_tokens or output_tokens:
                        tracker.record(self._agent_name, model, input_tokens, output_tokens)
        except Exception as e:
            logger.debug("Token tracking failed: %s", e)
