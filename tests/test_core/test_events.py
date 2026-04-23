"""Tests for src.core.events - EventStatus, AgentEvent, make_event helpers."""

from __future__ import annotations

import time

import pytest

from src.core.events import (
    AgentEvent,
    EventStatus,
    make_completed_event,
    make_event,
    make_failed_event,
    make_running_event,
    make_started_event,
)


class TestEventStatus:
    """Tests for the EventStatus enum."""

    def test_event_status_values(self):
        assert EventStatus.STARTED.value == "started"
        assert EventStatus.RUNNING.value == "running"
        assert EventStatus.COMPLETED.value == "completed"
        assert EventStatus.FAILED.value == "failed"

    def test_event_status_from_string(self):
        assert EventStatus("started") is EventStatus.STARTED
        assert EventStatus("running") is EventStatus.RUNNING
        assert EventStatus("completed") is EventStatus.COMPLETED
        assert EventStatus("failed") is EventStatus.FAILED

    def test_event_status_invalid_string(self):
        with pytest.raises(ValueError):
            EventStatus("unknown")

    def test_event_status_is_str(self):
        """EventStatus inherits from str."""
        assert isinstance(EventStatus.STARTED, str)
        assert EventStatus.STARTED == "started"

    def test_all_statuses(self):
        all_statuses = list(EventStatus)
        assert len(all_statuses) == 4


class TestAgentEvent:
    """Tests for the AgentEvent dataclass."""

    def test_agent_event_creation(self):
        evt = AgentEvent(
            agent="prospect",
            status=EventStatus.STARTED,
        )
        assert evt.agent == "prospect"
        assert evt.status is EventStatus.STARTED
        assert evt.step == ""
        assert evt.progress == 0
        assert evt.detail == ""
        assert isinstance(evt.timestamp, float)

    def test_agent_event_with_all_fields(self):
        evt = AgentEvent(
            agent="retention",
            status=EventStatus.RUNNING,
            step="feature_engineering",
            progress=50,
            detail="Building features...",
        )
        assert evt.agent == "retention"
        assert evt.step == "feature_engineering"
        assert evt.progress == 50
        assert evt.detail == "Building features..."

    def test_agent_event_to_dict(self):
        evt = AgentEvent(
            agent="ad",
            status=EventStatus.COMPLETED,
            step="done",
            progress=100,
        )
        d = evt.to_dict()
        assert d["agent"] == "ad"
        assert d["status"] == "completed"
        assert d["step"] == "done"
        assert d["progress"] == 100
        assert "timestamp" in d

    def test_agent_event_from_dict(self):
        data = {
            "agent": "subsidy",
            "status": "failed",
            "step": "optimization",
            "progress": 30,
            "detail": "OOM error",
        }
        evt = AgentEvent.from_dict(data)
        assert evt.agent == "subsidy"
        assert evt.status is EventStatus.FAILED
        assert evt.step == "optimization"

    def test_agent_event_from_dict_with_enum_status(self):
        data = {
            "agent": "prospect",
            "status": EventStatus.COMPLETED,
        }
        evt = AgentEvent.from_dict(data)
        assert evt.status is EventStatus.COMPLETED

    def test_roundtrip_dict(self):
        """to_dict -> from_dict roundtrip preserves data."""
        original = AgentEvent(
            agent="conversion",
            status=EventStatus.RUNNING,
            step="analysis",
            progress=75,
            detail="computing...",
        )
        restored = AgentEvent.from_dict(original.to_dict())
        assert restored.agent == original.agent
        assert restored.status == original.status
        assert restored.step == original.step
        assert restored.progress == original.progress
        assert restored.detail == original.detail


class TestMakeEvent:
    """Tests for the make_event helper and shorthands."""

    def test_make_event_basic(self):
        evt = make_event("prospect", EventStatus.STARTED)
        assert evt["agent"] == "prospect"
        assert evt["status"] == "started"

    def test_make_event_with_string_status(self):
        evt = make_event("prospect", "running")
        assert evt["status"] == "running"

    def test_make_event_with_kwargs(self):
        evt = make_event(
            "retention",
            EventStatus.RUNNING,
            step="training",
            progress=60,
            detail="Model training...",
        )
        assert evt["step"] == "training"
        assert evt["progress"] == 60
        assert evt["detail"] == "Model training..."

    def test_make_started_event(self):
        evt = make_started_event("ad", step="init", detail="Starting ad analysis")
        assert evt["status"] == "started"
        assert evt["agent"] == "ad"
        assert evt["step"] == "init"

    def test_make_running_event(self):
        evt = make_running_event("subsidy", step="causal", progress=40)
        assert evt["status"] == "running"
        assert evt["progress"] == 40

    def test_make_completed_event(self):
        evt = make_completed_event("prospect", step="done", detail="All good")
        assert evt["status"] == "completed"
        assert evt["progress"] == 100

    def test_make_failed_event(self):
        evt = make_failed_event("conversion", step="loading", detail="File not found")
        assert evt["status"] == "failed"
        assert evt["detail"] == "File not found"
