"""Tests for src.agent.tools — agent tool specs and dispatch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.tools import (
    build_tool_specs,
    execute_tool_call_async,
    get_spec_by_name,
    to_langchain_tools,
)


class TestToolSpecs:
    def test_seven_tools_exposed(self):
        specs = build_tool_specs()
        names = {s.name for s in specs}
        assert names == {
            "run_prospect_analysis",
            "run_conversion_analysis",
            "run_subsidy_analysis",
            "run_retention_analysis",
            "run_ad_analysis",
            "run_budget_analysis",
            "search_memory",
        }

    def test_langchain_tools_render_schemas(self):
        tools = to_langchain_tools(build_tool_specs())
        assert len(tools) == 7
        by_name = {t.name: t for t in tools}
        assert "query" in by_name["run_prospect_analysis"].args_schema.model_fields
        # subsidy takes budget, memory search does not
        assert "budget" in by_name["run_subsidy_analysis"].args_schema.model_fields
        assert "budget" not in by_name["search_memory"].args_schema.model_fields

    def test_get_spec_by_name(self):
        assert get_spec_by_name("search_memory") is not None
        assert get_spec_by_name("nope") is None


class TestExecuteToolCall:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        result = await execute_tool_call_async({}, "does_not_exist", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_expert_tool_runs_full_lifecycle(self):
        """An expert tool call routes through BaseAgent.run() and returns a digest."""
        from src.graph.state import AnalysisStatus  # noqa: F401 (import sanity)

        fake_expert = MagicMock()
        fake_expert.run = AsyncMock(
            return_value={
                "success": True,
                "kpis": {"expected_roi": 2.1},
                "analysis": {"summary": "补贴ROI健康"},
            }
        )
        state = {"budget": 50000, "data_path": "", "org_id": "org-1"}

        with (
            patch("src.graph.nodes._create_expert", return_value=fake_expert),
            patch("src.graph.nodes._validate_expert_result", side_effect=lambda k, r: r),
        ):
            result = await execute_tool_call_async(
                state, "run_subsidy_analysis", {"query": "分析补贴", "budget": 50000}
            )

        fake_expert.run.assert_awaited_once()
        call_state = fake_expert.run.await_args[0][0]
        assert call_state["params"]["budget"] == 50000
        assert call_state["params"]["scope"] == "subsidy"
        # digest carries the compact view + full result for state merge
        assert result["kpis"] == {"expected_roi": 2.1}
        assert result["_full_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_memory_search_never_raises(self):
        """Memory unavailable degrades to found=0, not an exception."""
        with patch(
            "src.memory.store.MemoryStore.search_analyses", side_effect=RuntimeError("no fs")
        ):
            result = await execute_tool_call_async(
                {"org_id": "o"}, "search_memory", {"query": "补贴"}
            )
        assert result["found"] == 0


class TestACIToolSpecs:
    def test_descriptions_are_self_describing(self):
        from src.agent.tools import build_tool_specs

        for spec in build_tool_specs():
            assert "用途" in spec.description or "检索" in spec.description
            if spec.name.startswith("run_"):
                assert "失败模式" in spec.description, f"{spec.name} must document failure modes"
                assert spec.failure_modes, f"{spec.name} needs structured failure_modes"

    def test_subsidy_documents_budget_requirement(self):
        from src.agent.tools import build_tool_specs

        subsidy = next(s for s in build_tool_specs() if s.name == "run_subsidy_analysis")
        assert "budget" in subsidy.description


class TestAdDataHonesty:
    def test_ad_result_marks_synthetic_source(self):
        from src.agents.ad import AdExpert

        result = AdExpert(llm=None)._execute_pipeline({"query": "广告", "budget": 0})
        assert result.get("data_source") == "synthetic"


class TestBudgetToolSpec:
    """Budget chief-tool registration + context_kpis plumbing."""

    def test_budget_tool_registered_with_failure_modes(self):
        from src.agent.tools import EXPERT_TOOL_FAILURE_MODES, get_spec_by_name

        spec = get_spec_by_name("run_budget_analysis")
        assert spec is not None
        assert "budget" in EXPERT_TOOL_FAILURE_MODES
        assert "跨渠道" in spec.description

    @pytest.mark.asyncio
    async def test_budget_tool_dispatch_carries_context_kpis(self, monkeypatch):
        """context_kpis (from kpi_snapshot) must reach the expert params."""
        from src.agent import tools as agent_tools

        captured = {}

        async def _capture_run(state):
            captured.update(state["params"])
            return {
                "success": True,
                "expert": "budget",
                "kpis": {"expected_roi": 1.5},
                "allocation": {"ad": 1.0},
            }

        fake = MagicMock()
        fake.run = AsyncMock(side_effect=_capture_run)
        monkeypatch.setattr("src.graph.nodes._create_expert", lambda key: fake)

        state = {
            "query": "全局预算分配",
            "budget": 100_000,
            "kpi_snapshot": {"subsidy_expected_roi": 2.0},
        }
        digest = await agent_tools.execute_tool_call_async(
            state,
            "run_budget_analysis",
            {"query": "全局预算分配", "budget": 100_000},
        )
        assert captured["context_kpis"] == {"subsidy_expected_roi": 2.0}
        assert digest["success"] is True
