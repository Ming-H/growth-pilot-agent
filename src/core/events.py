"""Event definitions for agent execution tracking.

Provides:
- EventStatus enum: STARTED / RUNNING / COMPLETED / FAILED
- AgentEvent dataclass: structured event with agent, status, step, progress, detail, timestamp
- make_event() helper: quick event creation with sensible defaults
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventStatus(str, Enum):
    """Agent execution status lifecycle."""

    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentEvent:
    """Structured event emitted during agent execution.

    Events are accumulated in ``AgentState.events`` via the ``operator.add``
    reducer, providing a chronological event stream for tracing and monitoring.

    Attributes:
        agent: Agent name (e.g. "prospect", "orchestrator").
        status: Current execution status.
        step: Descriptive step name (e.g. "feature_engineering", "llm_synthesis").
        progress: Progress percentage 0-100.
        detail: Human-readable detail message.
        timestamp: Unix timestamp of event creation.
    """

    agent: str
    status: EventStatus
    step: str = ""
    progress: int = 0
    detail: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for AgentState.events."""
        return {
            "agent": self.agent,
            "status": self.status.value,
            "step": self.step,
            "progress": self.progress,
            "detail": self.detail,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentEvent:
        """Deserialize from a plain dict."""
        status_val = data.get("status", "started")
        if isinstance(status_val, str):
            status_val = EventStatus(status_val)
        return cls(
            agent=data.get("agent", ""),
            status=status_val,
            step=data.get("step", ""),
            progress=data.get("progress", 0),
            detail=data.get("detail", ""),
            timestamp=data.get("timestamp", time.time()),
        )


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def make_event(
    agent: str,
    status: EventStatus | str = EventStatus.STARTED,
    *,
    step: str = "",
    progress: int = 0,
    detail: str = "",
) -> dict[str, Any]:
    """Quick-create an event and return it as a dict for state injection.

    Usage::

        events = [
            make_event("prospect", EventStatus.STARTED, step="init"),
            make_event("prospect", EventStatus.RUNNING, step="feature_eng", progress=30),
            make_event("prospect", EventStatus.COMPLETED, step="done", progress=100),
        ]
        return {"events": events}
    """
    if isinstance(status, str):
        status = EventStatus(status)
    return AgentEvent(
        agent=agent,
        status=status,
        step=step,
        progress=progress,
        detail=detail,
    ).to_dict()


def make_started_event(agent: str, step: str = "init", detail: str = "") -> dict[str, Any]:
    """Shorthand for a STARTED event."""
    return make_event(agent, EventStatus.STARTED, step=step, detail=detail)


def make_running_event(agent: str, step: str = "", progress: int = 0, detail: str = "") -> dict[str, Any]:
    """Shorthand for a RUNNING event."""
    return make_event(agent, EventStatus.RUNNING, step=step, progress=progress, detail=detail)


def make_completed_event(agent: str, step: str = "done", detail: str = "") -> dict[str, Any]:
    """Shorthand for a COMPLETED event."""
    return make_event(agent, EventStatus.COMPLETED, step=step, progress=100, detail=detail)


def make_failed_event(agent: str, step: str = "", detail: str = "") -> dict[str, Any]:
    """Shorthand for a FAILED event."""
    return make_event(agent, EventStatus.FAILED, step=step, detail=detail)
