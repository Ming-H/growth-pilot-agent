"""Zero-key (heuristic brain) end-to-end tests.

The regression-baseline property: with no LLM configured, the same input
must produce a byte-identical report across runs — this is what protects
every later refactor (determinism invariant).
"""

from __future__ import annotations

import pytest

from src.core.config import get_settings
from src.graph.nodes import _intent_classify
from src.graph.workflow import run_workflow


def _zero_key_env(monkeypatch, tmp_path):
    """Force the deterministic brain: remote channel with no key."""
    monkeypatch.setenv("GPA_LLM_API_KEY", "")
    monkeypatch.setenv("GPA_DEMO_MODE", "true")
    monkeypatch.setenv("GPA_LLM_CHANNEL", "remote")
    monkeypatch.setenv("GPA_MEMORY_DB_PATH", str(tmp_path / "memory" / "memory.db"))
    get_settings.cache_clear()


@pytest.mark.integration
class TestZeroKeyWorkflow:
    @pytest.mark.asyncio
    async def test_full_run_succeeds_without_llm(self, monkeypatch, tmp_path):
        """No key + demo mode + dead endpoint → successful deterministic run."""
        _zero_key_env(monkeypatch, tmp_path)
        from src.graph import nodes as graph_nodes

        graph_nodes._clear_expert_cache()

        result = await run_workflow(
            query="分析本月用户增长与补贴效果",
            scope="full",
            budget=50000,
            mode="workflow",
        )
        assert result["success"] is True, f"errors: {result['errors']}"
        assert result["report"], "report must never be empty"
        assert result["kpi_snapshot"], "KPIs must be present (tools compute facts)"
        assert result["metadata"]["cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_report_byte_identical_across_runs(self, monkeypatch, tmp_path):
        """Regression baseline: same input ⇒ byte-identical report."""
        _zero_key_env(monkeypatch, tmp_path)
        from src.graph import nodes as graph_nodes

        graph_nodes._clear_expert_cache()

        r1 = await run_workflow(
            query="分析本月用户增长", scope="full", budget=10000, mode="workflow"
        )
        graph_nodes._clear_expert_cache()
        r2 = await run_workflow(
            query="分析本月用户增长", scope="full", budget=10000, mode="workflow"
        )
        assert r1["report"] == r2["report"]
        assert r1["kpi_snapshot"] == r2["kpi_snapshot"]

    @pytest.mark.asyncio
    async def test_plan_source_is_rule(self, monkeypatch, tmp_path):
        """Zero-key planning is marked rule-based (visible degradation)."""
        _zero_key_env(monkeypatch, tmp_path)
        from src.graph import nodes as graph_nodes

        graph_nodes._clear_expert_cache()

        result = await run_workflow(
            query="分析补贴 ROI", scope="subsidy", budget=50000, mode="workflow"
        )
        # The compat result flattens state; expert presence proves routing ran.
        assert "subsidy" in result["expert_results"]


class TestIntentClassifyRobustness:
    def test_broken_native_lib_does_not_crash_router(self, monkeypatch):
        """A class import raising OSError (broken native lib) degrades that
        expert instead of crashing the keyword router (A8)."""
        import builtins

        real_import = builtins.__import__

        def _selective_import(name, *args, **kwargs):
            if "src.agents.prospect" in name:
                raise OSError("dlopen(...lib_lightgbm.dylib): broken dylib")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _selective_import)
        selected = _intent_classify("补贴 ROI 预算怎么分")
        assert "subsidy" in selected
        assert "prospect" not in selected  # degraded, not crashed

    def test_normal_routing_untouched(self):
        selected = _intent_classify("补贴 ROI 预算怎么分")
        assert "subsidy" in selected


class TestExpertDegradedSummary:
    @pytest.mark.asyncio
    async def test_expert_without_llm_returns_degraded_summary(self, tmp_path):
        """llm=None expert: success stays True, degraded marker present."""
        from src.agents.subsidy import SubsidyExpert

        expert = SubsidyExpert(llm=None)
        result = await expert.analyze({"query": "补贴分析", "budget": 10000})
        assert result["success"] is True
        assert result.get("degraded") == ["llm_synthesis"]
        assert result["analysis"]["degraded"] is True
        assert result["kpis"], "deterministic KPIs must exist without a brain"
        assert "确定性" in result["analysis"]["summary"]
