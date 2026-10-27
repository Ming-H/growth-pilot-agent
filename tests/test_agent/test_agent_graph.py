"""Tests for the agent-led StateGraph: routing, gate, and end-to-end loop."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.chief import (
    _last_ai_text,
    _route_after_approval,
    agent_tools_node,
    evaluate_gate_node,
    should_continue,
)
from src.graph.state import AnalysisStatus


def _agent_state(**overrides) -> dict:
    base = {
        "query": "分析补贴预算怎么分",
        "org_id": "org-1",
        "user_id": "u-1",
        "scope": "subsidy",
        "budget": 50000.0,
        "data_path": "",
        "expert_results": [],
        "execution_errors": [],
        "refinement_round": 0,
        "messages": [],
    }
    base.update(overrides)
    return base


class TestRouting:
    def test_continue_when_tool_calls_present(self):
        state = _agent_state(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "run_subsidy_analysis", "args": {}, "id": "1"}],
                )
            ]
        )
        assert should_continue(state) == "tools"

    def test_gate_when_no_tool_calls(self):
        state = _agent_state(messages=[AIMessage(content="最终报告...")])
        assert should_continue(state) == "evaluate"

    def test_approval_routing(self):
        assert _route_after_approval({"approved": True}) == "report"
        assert _route_after_approval({"approved": False}) == "__end__"
        assert _route_after_approval({}) == "__end__"


class TestToolsNode:
    @pytest.mark.asyncio
    async def test_tool_calls_merged_into_state(self):
        digest = {
            "expert": "subsidy",
            "success": True,
            "kpis": {"expected_roi": 2.0},
            "_full_result": {
                "expert": "subsidy",
                "success": True,
                "kpis": {"expected_roi": 2.0},
                "ate": {"ate": 0.05},
            },
        }
        state = _agent_state(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "run_subsidy_analysis",
                            "args": {"query": "补贴", "budget": 50000},
                            "id": "call_1",
                        }
                    ],
                )
            ]
        )
        with patch("src.agent.tools.execute_tool_call_async", new=AsyncMock(return_value=digest)):
            update = await agent_tools_node(state)

        # ToolMessage returned to the LLM; full result merged for state
        assert len(update["messages"]) == 1
        assert isinstance(update["messages"][0], ToolMessage)
        assert update["messages"][0].tool_call_id == "call_1"
        assert "_full_result" not in json.loads(update["messages"][0].content)
        assert update["expert_results"][0]["ate"] == {"ate": 0.05}

    @pytest.mark.asyncio
    async def test_failed_expert_recorded_as_error(self):
        digest = {
            "expert": "ad",
            "success": False,
            "error": "boom",
            "_full_result": {"expert": "ad", "success": False, "error": "boom"},
        }
        state = _agent_state(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "run_ad_analysis", "args": {"query": "ad"}, "id": "c2"}],
                )
            ]
        )
        with patch("src.agent.tools.execute_tool_call_async", new=AsyncMock(return_value=digest)):
            update = await agent_tools_node(state)
        assert update["execution_errors"] == ["ad: boom"]


class TestQualityGate:
    @pytest.mark.asyncio
    async def test_low_score_injects_feedback(self):
        from src.core.evaluator import QualityScore

        low = QualityScore(
            completeness=0.4,
            actionability=0.5,
            data_grounding=0.6,
            overall=0.5,
            reasoning="缺少数据支撑",
        )
        state = _agent_state(messages=[HumanMessage(content="q"), AIMessage(content="报告")])
        with patch("src.core.evaluator.evaluate_expert_output", new=AsyncMock(return_value=low)):
            update = await evaluate_gate_node(state)

        assert update["needs_refinement"] is True
        assert update["refinement_round"] == 1
        assert isinstance(update["messages"][0], HumanMessage)
        assert "缺少数据支撑" in update["messages"][0].content

    @pytest.mark.asyncio
    async def test_high_score_accepts(self):
        from src.core.evaluator import QualityScore

        high = QualityScore(completeness=0.9, actionability=0.85, data_grounding=0.9, overall=0.88)
        state = _agent_state(messages=[AIMessage(content="报告")])
        with patch("src.core.evaluator.evaluate_expert_output", new=AsyncMock(return_value=high)):
            update = await evaluate_gate_node(state)
        assert update["needs_refinement"] is False
        assert "messages" not in update


class TestLastAiText:
    def test_prefers_non_tool_message(self):
        state = _agent_state(
            messages=[
                AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
                ToolMessage(content="r", tool_call_id="1"),
                AIMessage(content="最终答案"),
            ]
        )
        assert _last_ai_text(state) == "最终答案"


class TestGraphEndToEnd:
    @pytest.mark.asyncio
    async def test_full_agent_loop_with_mocks(self):
        """agent → tools → agent (final) → evaluate → approval → report."""
        from src.agent.graph import build_agent_graph

        llm_responses = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "run_subsidy_analysis",
                            "args": {"query": "补贴怎么分", "budget": 50000},
                            "id": "c1",
                        }
                    ],
                ),
                AIMessage(content="# 补贴分析报告\n结论：ROI 2.1x"),
            ]
        )
        fake_bound = MagicMock()
        fake_bound.ainvoke = AsyncMock(side_effect=lambda msgs: next(llm_responses))
        fake_llm = MagicMock()
        fake_llm.bind_tools = MagicMock(return_value=fake_bound)

        digest = {
            "expert": "subsidy",
            "success": True,
            "kpis": {"expected_roi": 2.1},
            "_full_result": {
                "expert": "subsidy",
                "success": True,
                "kpis": {"expected_roi": 2.1},
                "analysis": {"summary": "ok"},
            },
        }

        from src.core.evaluator import QualityScore

        high = QualityScore(completeness=0.9, actionability=0.9, data_grounding=0.9, overall=0.9)

        with (
            patch("src.core.llm_factory.create_llm", return_value=fake_llm),
            patch(
                "src.agent.tools.execute_tool_call_async",
                new=AsyncMock(return_value=digest),
            ),
            patch(
                "src.core.evaluator.evaluate_expert_output",
                new=AsyncMock(return_value=high),
            ),
            patch(
                "src.guardrails.output_guard.validate_output",
                return_value=MagicMock(passed=True),
            ),
            patch(
                "src.memory.store.MemoryStore.save_analysis",
                new=AsyncMock(),
            ) as mock_store,
        ):
            graph = build_agent_graph()
            compiled = graph.compile()
            final = await compiled.ainvoke(_agent_state(approval_required=False))

        assert final["status"] == AnalysisStatus.COMPLETED
        assert "ROI" in final["final_report"]
        assert final["expert_results"][0]["expert"] == "subsidy"
        # memory write-back happened
        mock_store.assert_called()


class TestModeDispatch:
    @pytest.mark.asyncio
    async def test_run_workflow_defaults_to_agent_mode(self):
        """run_workflow dispatches to the agent graph when mode is unset."""
        from src.graph import workflow as wf

        agent_result = {
            "status": AnalysisStatus.COMPLETED,
            "final_report": "R",
            "expert_results": [],
            "execution_errors": [],
        }
        with (
            patch(
                "src.agent.graph.run_agent_analysis", new=AsyncMock(return_value=agent_result)
            ) as mock_agent,
            patch("src.graph.graph.run_analysis", new=AsyncMock()) as mock_wf,
        ):
            out = await wf.run_workflow(query="分析补贴", budget=1000, scope="subsidy")

        mock_agent.assert_awaited_once()
        mock_wf.assert_not_awaited()
        assert out["report"] == "R"
        assert out["success"] is True

    @pytest.mark.asyncio
    async def test_run_workflow_explicit_workflow_mode(self):
        from src.graph import workflow as wf

        wf_result = {
            "status": AnalysisStatus.COMPLETED,
            "final_report": "W",
            "expert_results": [],
            "execution_errors": [],
        }
        with (
            patch("src.agent.graph.run_agent_analysis", new=AsyncMock()) as mock_agent,
            patch("src.graph.graph.run_analysis", new=AsyncMock(return_value=wf_result)) as mock_wf,
        ):
            out = await wf.run_workflow(query="q" * 10, mode="workflow")

        mock_wf.assert_awaited_once()
        mock_agent.assert_not_awaited()
        assert out["report"] == "W"
