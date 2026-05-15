"""Observability integrations for GrowthPilot Agent.

Supports LangSmith tracing for Multi-Agent execution visualization
alongside existing OpenTelemetry instrumentation, cost tracking,
and distributed tracing.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextvars import ContextVar
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("growth-pilot-agent")

# Context variable for distributed tracing across graph nodes
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def setup_telemetry(service_name: str = "growth-pilot-agent", enabled: bool = True) -> None:
    """Initialize OpenTelemetry tracing."""
    if not enabled:
        return
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing initialized for %s", service_name)


def setup_langsmith(
    *,
    enabled: bool = False,
    api_key: str = "",
    project: str = "growth-pilot",
) -> None:
    """Configure LangSmith tracing for agent execution visualization.

    Sets environment variables that LangChain/LangGraph automatically
    pick up for trace collection.

    Args:
        enabled: Whether to enable LangSmith tracing.
        api_key: LangSmith API key (or set LANGSMITH_API_KEY env var).
        project: LangSmith project name for organizing traces.
    """
    if not enabled:
        logger.debug("[observability] LangSmith tracing disabled")
        return

    if not api_key:
        logger.warning("[observability] LangSmith enabled but no API key provided")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    logger.info("[observability] LangSmith tracing enabled (project=%s)", project)


def setup_otel(
    *,
    enabled: bool = False,
    service_name: str = "growth-pilot-agent",
    endpoint: str = "",
) -> None:
    """Configure OpenTelemetry instrumentation.

    Args:
        enabled: Whether to enable OTel instrumentation.
        service_name: Service name for traces.
        endpoint: OTel collector endpoint (e.g. "http://localhost:4317").
    """
    if not enabled:
        return

    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=endpoint)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        logger.info("[observability] OpenTelemetry enabled (service=%s)", service_name)
    except ImportError:
        logger.warning("[observability] OpenTelemetry packages not installed, skipping")
    except Exception as exc:
        logger.warning("[observability] OTel setup failed: %s", exc)


def setup_observability(settings: Any) -> None:
    """Set up all observability backends from application settings.

    Args:
        settings: Application Settings instance (from src.core.config).
    """
    setup_langsmith(
        enabled=getattr(settings, "langsmith_enabled", False),
        api_key=getattr(settings, "langsmith_api_key", ""),
        project=getattr(settings, "langsmith_project", "growth-pilot"),
    )

    setup_otel(
        enabled=getattr(settings, "otel_enabled", False),
        service_name=getattr(settings, "otel_service_name", "growth-pilot-agent"),
    )


class CostTracker:
    """Track token usage and estimated cost per agent per run."""

    # Pricing per 1M tokens (as of 2024)
    PRICING: dict[str, dict[str, float]] = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "glm-4.7": {"input": 0.000001, "output": 0.000002},
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
