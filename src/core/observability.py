"""Observability: OpenTelemetry tracing and cost tracking."""
from __future__ import annotations

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("growth-pilot-agent")


def setup_telemetry(service_name: str = "growth-pilot-agent", enabled: bool = True) -> None:
    """Initialize OpenTelemetry tracing."""
    if not enabled:
        return
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info(f"OpenTelemetry tracing initialized for {service_name}")


class CostTracker:
    """Track token usage and estimated cost per agent per run."""

    # Pricing per 1M tokens (as of 2024)
    PRICING: dict[str, dict[str, float]] = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "deepseek-chat": {"input": 0.14, "output": 0.28},
    }

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record token usage for a single LLM call."""
        prices = self.PRICING.get(model, {"input": 0.50, "output": 1.50})
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
        self._records.append({
            "agent": agent,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "timestamp": time.time(),
        })

    @property
    def total_cost(self) -> float:
        return round(sum(r["cost_usd"] for r in self._records), 6)

    @property
    def total_tokens(self) -> int:
        return sum(r["input_tokens"] + r["output_tokens"] for r in self._records)

    def report(self) -> dict[str, Any]:
        """Generate a cost summary report."""
        by_agent: dict[str, dict[str, Any]] = defaultdict(lambda: {"cost_usd": 0.0, "calls": 0, "tokens": 0})
        for r in self._records:
            by_agent[r["agent"]]["cost_usd"] += r["cost_usd"]
            by_agent[r["agent"]]["calls"] += 1
            by_agent[r["agent"]]["tokens"] += r["input_tokens"] + r["output_tokens"]

        return {
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_calls": len(self._records),
            "by_agent": dict(by_agent),
            "records": self._records,
        }

    def reset(self) -> None:
        """Clear all records."""
        self._records.clear()


def traced(name: str):
    """Decorator to wrap an async function with an OTEL span."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("function", func.__name__)
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("status", "ok")
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e)[:200])
                    raise
        return wrapper
    return decorator
