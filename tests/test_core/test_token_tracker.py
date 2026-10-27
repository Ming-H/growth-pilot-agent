"""Tests for run-scoped cost attribution (ContextVar-based CostTracker)."""

from __future__ import annotations

import asyncio

import pytest

from src.core.token_tracker import (
    CostTracker,
    cost_scope,
    get_cost_tracker,
    run_cost_tracker,
)

pytestmark = pytest.mark.asyncio


class TestCostScope:
    async def test_records_isolated_per_scope(self):
        async def _run(agent: str) -> dict:
            async with cost_scope() as tracker:
                tracker.record(agent, "glm-4.7", 100, 50)
                await asyncio.sleep(0)  # yield to interleave
                return tracker.report()

        r1, r2 = await asyncio.gather(_run("agent_one"), _run("agent_two"))
        assert r1["total_calls"] == 1
        assert r1["by_agent"]["agent_one"]["calls"] == 1
        assert r2["total_calls"] == 1
        assert r2["by_agent"]["agent_two"]["calls"] == 1

    async def test_to_thread_propagates_scope(self):
        """Worker-thread records (the expert pipeline path) land in the run scope."""
        async with cost_scope() as tracker:
            await asyncio.to_thread(self._record_via_thread, "thread_agent")
            assert tracker.report()["by_agent"]["thread_agent"]["calls"] == 1

    @staticmethod
    def _record_via_thread(agent: str) -> None:
        tracker = run_cost_tracker.get()
        assert tracker is not None
        tracker.record(agent, "glm-4.7", 10, 5)

    async def test_outside_scope_detached_tracker(self):
        """get_cost_tracker outside a scope yields a fresh (zero) tracker,
        never another run's records."""
        tracker = get_cost_tracker()
        assert tracker.report()["total_calls"] == 0

    async def test_scope_exit_resets(self):
        async with cost_scope() as tracker:
            tracker.record("a", "glm-4.7", 1, 1)
        assert run_cost_tracker.get() is None


class TestCostTracker:
    def test_report_shape(self):
        t = CostTracker()
        t.record("chief", "glm-4.7", 1000, 500)
        report = t.report()
        assert report["total_calls"] == 1
        assert report["input_tokens"] == 1000
        assert report["output_tokens"] == 500
        assert report["total_tokens"] == 1500
        assert report["by_agent"]["chief"]["calls"] == 1
        assert isinstance(report["total_cost_usd"], float)
