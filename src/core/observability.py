"""Observability integrations for GrowthPilot Agent.

Supports LangSmith tracing for Multi-Agent execution visualization
alongside existing OpenTelemetry instrumentation, cost tracking,
and distributed tracing.
"""

from __future__ import annotations

import logging
import os
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


# NOTE: cost tracking lives in src/core/token_tracker.py (single home — the
# parallel CostTracker that used to live here was dead code, removed in v9).


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
