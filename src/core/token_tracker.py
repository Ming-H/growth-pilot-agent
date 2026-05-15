"""Token usage tracking via LangChain callback handler."""
from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

_tracker_instance: CostTracker | None = None
_tracker_lock = threading.Lock()


class CostTracker:
    """Track token usage and estimated cost per agent per run."""

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
        self._records.append({
            "agent": agent,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })

    @property
    def total_cost(self) -> float:
        return round(sum(r["cost_usd"] for r in self._records), 6)

    @property
    def total_tokens(self) -> int:
        return sum(r["input_tokens"] + r["output_tokens"] for r in self._records)

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
            "total_calls": len(self._records),
            "by_agent": dict(by_agent),
        }

    def reset(self) -> None:
        self._records.clear()


def get_cost_tracker() -> CostTracker:
    """Return the singleton CostTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = CostTracker()
    return _tracker_instance


def reset_cost_tracker() -> None:
    """Reset the singleton for a new analysis run."""
    get_cost_tracker().reset()


class TokenTrackingCallback(BaseCallbackHandler):
    """LangChain callback handler that records token usage to CostTracker."""

    def __init__(self, agent_name: str = "unknown") -> None:
        self._agent_name = agent_name

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        tracker = get_cost_tracker()
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
                                model = response_metadata.get(
                                    "model_name", fallback_model
                                )
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
