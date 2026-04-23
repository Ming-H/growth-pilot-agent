"""Tests for the agent lifecycle hooks (observability layer)."""
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
from src.core.hooks import LoggingHook, MetricsHook, TracingHook


class TestEventStatus:
    """Tests for the EventStatus enum."""

    def test_status_values(self):
        """EventStatus has the expected string values."""
        assert EventStatus.STARTED == "started"
        assert EventStatus.RUNNING == "running"
        assert EventStatus.COMPLETED == "completed"
        assert EventStatus.FAILED == "failed"


class TestAgentEvent:
    """Tests for the AgentEvent dataclass."""

    def test_basic_creation(self):
        """AgentEvent can be created with required fields."""
        evt = AgentEvent(agent="prospect", status=EventStatus.STARTED)
        assert evt.agent == "prospect"
        assert evt.status == EventStatus.STARTED
        assert evt.step == ""
        assert evt.progress == 0

    def test_full_creation(self):
        """AgentEvent accepts all optional fields."""
        evt = AgentEvent(
            agent="conversion",
            status=EventStatus.RUNNING,
            step="funnel_analysis",
            progress=50,
            detail="Processing funnel data",
        )
        assert evt.step == "funnel_analysis"
        assert evt.progress == 50
        assert evt.detail == "Processing funnel data"

    def test_to_dict(self):
        """to_dict returns a plain dict with expected keys."""
        evt = AgentEvent(agent="test", status=EventStatus.COMPLETED, step="done")
        d = evt.to_dict()
        assert d["agent"] == "test"
        assert d["status"] == "completed"
        assert d["step"] == "done"
        assert "timestamp" in d

    def test_from_dict(self):
        """from_dict reconstructs an AgentEvent from a plain dict."""
        data = {
            "agent": "retention",
            "status": "running",
            "step": "churn",
            "progress": 30,
            "detail": "Predicting churn",
        }
        evt = AgentEvent.from_dict(data)
        assert evt.agent == "retention"
        assert evt.status == EventStatus.RUNNING
        assert evt.step == "churn"
        assert evt.progress == 30

    def test_from_dict_with_enum_status(self):
        """from_dict handles EventStatus enum values in the dict."""
        data = {"agent": "ad", "status": EventStatus.FAILED}
        evt = AgentEvent.from_dict(data)
        assert evt.status == EventStatus.FAILED

    def test_roundtrip_dict(self):
        """to_dict -> from_dict roundtrip preserves data."""
        original = AgentEvent(
            agent="subsidy",
            status=EventStatus.COMPLETED,
            step="alloc",
            progress=100,
            detail="Done",
        )
        restored = AgentEvent.from_dict(original.to_dict())
        assert restored.agent == original.agent
        assert restored.status == original.status
        assert restored.step == original.step
        assert restored.progress == original.progress
        assert restored.detail == original.detail


class TestMakeEvent:
    """Tests for the make_event helper and convenience functions."""

    def test_make_event_started(self):
        """make_event with STARTED status."""
        d = make_event("prospect", EventStatus.STARTED, step="init")
        assert d["agent"] == "prospect"
        assert d["status"] == "started"
        assert d["step"] == "init"

    def test_make_event_with_string_status(self):
        """make_event accepts a string for the status argument."""
        d = make_event("ad", "running", progress=40)
        assert d["status"] == "running"
        assert d["progress"] == 40

    def test_make_started_event(self):
        """make_started_event helper creates a started event dict."""
        d = make_started_event("conversion", step="load", detail="Loading data")
        assert d["status"] == "started"
        assert d["agent"] == "conversion"
        assert d["step"] == "load"
        assert d["detail"] == "Loading data"

    def test_make_running_event(self):
        """make_running_event helper creates a running event dict."""
        d = make_running_event("retention", step="predict", progress=60)
        assert d["status"] == "running"
        assert d["progress"] == 60

    def test_make_completed_event(self):
        """make_completed_event helper creates a completed event with progress=100."""
        d = make_completed_event("subsidy", step="done", detail="All good")
        assert d["status"] == "completed"
        assert d["progress"] == 100
        assert d["detail"] == "All good"

    def test_make_failed_event(self):
        """make_failed_event helper creates a failed event."""
        d = make_failed_event("ad", step="bid", detail="Timeout")
        assert d["status"] == "failed"
        assert d["detail"] == "Timeout"


class TestTracingHook:
    """Tests for the TracingHook (timing observability)."""

    @pytest.mark.asyncio
    async def test_pre_run_injects_start_time(self):
        """TracingHook.on_pre_run injects _trace_start into state."""
        hook = TracingHook()
        state = {}
        result = await hook.on_pre_run("prospect", state)
        assert "_trace_start" in result

    @pytest.mark.asyncio
    async def test_post_run_records_elapsed(self):
        """TracingHook.on_post_run records elapsed time in result."""
        hook = TracingHook()
        state = {"_trace_start": time.time() - 1.0}  # 1 second ago
        result = {}
        updated = await hook.on_post_run("prospect", result, state)
        assert "_trace" in updated
        assert updated["_trace"]["agent"] == "prospect"
        assert updated["_trace"]["elapsed_s"] >= 0.9

    @pytest.mark.asyncio
    async def test_post_run_without_start(self):
        """TracingHook.on_post_run handles missing _trace_start gracefully."""
        hook = TracingHook()
        result = {}
        state = {}
        updated = await hook.on_post_run("ad", result, state)
        assert "_trace" in updated
        assert updated["_trace"]["elapsed_s"] >= 0


class TestLoggingHook:
    """Tests for the LoggingHook."""

    @pytest.mark.asyncio
    async def test_pre_run_returns_state(self):
        """LoggingHook.on_pre_run returns the state unchanged."""
        hook = LoggingHook()
        state = {"query": "test"}
        result = await hook.on_pre_run("prospect", state)
        assert result == state

    @pytest.mark.asyncio
    async def test_post_run_no_errors(self):
        """LoggingHook.on_post_run with no errors returns result unchanged."""
        hook = LoggingHook()
        result = {"data": "value"}
        state = {}
        updated = await hook.on_post_run("ad", result, state)
        assert updated == result

    @pytest.mark.asyncio
    async def test_post_run_with_errors(self):
        """LoggingHook.on_post_run with errors still returns result."""
        hook = LoggingHook()
        result = {"data": "value", "errors": ["something broke"]}
        state = {}
        updated = await hook.on_post_run("ad", result, state)
        assert updated == result


class TestMetricsHook:
    """Tests for the MetricsHook."""

    @pytest.mark.asyncio
    async def test_post_run_adds_metrics(self):
        """MetricsHook.on_post_run injects _metrics into the result."""
        hook = MetricsHook()
        result = {}
        state = {}
        updated = await hook.on_post_run("prospect", result, state)
        assert "_metrics" in updated
        assert updated["_metrics"]["agent"] == "prospect"
        assert updated["_metrics"]["has_errors"] is False
        assert "timestamp" in updated["_metrics"]

    @pytest.mark.asyncio
    async def test_post_run_detects_errors(self):
        """MetricsHook correctly detects errors in result."""
        hook = MetricsHook()
        result = {"errors": ["fail"]}
        state = {}
        updated = await hook.on_post_run("conversion", result, state)
        assert updated["_metrics"]["has_errors"] is True
