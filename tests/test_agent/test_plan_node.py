"""Tests for the v9 planning pillar and the KPI reconciliation gate."""

from __future__ import annotations

import pytest

from src.agent.chief import (
    _heuristic_brain_message,
    _validate_subgoals,
    agent_plan_node,
)


class TestSubgoalValidator:
    """只删不造 (only-delete-never-create) — the planner contract."""

    def test_whitelist_filter(self):
        raw = {
            "subgoals": [
                {"expert": "subsidy", "question": "补贴 ROI"},
                {"expert": "browser", "question": "上网搜"},  # not whitelisted → dropped
            ]
        }
        out = _validate_subgoals(raw, "q", None)
        assert [sg["expert"] for sg in out] == ["subsidy"]

    def test_dedupe_per_expert(self):
        raw = {
            "subgoals": [
                {"expert": "ad", "question": "one"},
                {"expert": "ad", "question": "two"},  # duplicate expert → dropped
            ]
        }
        out = _validate_subgoals(raw, "q", None)
        assert len(out) == 1

    def test_cap_five(self):
        raw = {
            "subgoals": [
                {"expert": e, "question": "q"}
                for e in ["prospect", "conversion", "subsidy", "retention", "ad"]
            ]
        }
        raw["subgoals"].append({"expert": "prospect", "question": "x"})  # dup anyway
        out = _validate_subgoals(raw, "q", None)
        assert len(out) <= 5

    def test_empty_question_dropped(self):
        raw = {"subgoals": [{"expert": "ad", "question": "   "}]}
        assert _validate_subgoals(raw, "q", None) == []

    def test_never_invents(self):
        """The validator cannot ADD experts the LLM did not name."""
        raw = {"subgoals": [{"expert": "bogus", "question": "q"}]}
        assert _validate_subgoals(raw, "任何查询", None) == []

    def test_budget_only_for_subsidy(self):
        raw = {"subgoals": [{"expert": "prospect", "question": "q", "budget": 500}]}
        out = _validate_subgoals(raw, "q", 500)
        assert "budget" not in out[0]

    def test_malformed_input_returns_empty(self):
        assert _validate_subgoals(None, "q", None) == []
        assert _validate_subgoals("not a dict", "q", None) == []
        assert _validate_subgoals({"subgoals": "nope"}, "q", None) == []


class TestPlanNodeRuleFallback:
    @pytest.mark.asyncio
    async def test_zero_key_uses_keyword_prior_with_marker(self, monkeypatch):
        monkeypatch.setenv("GPA_LLM_API_KEY", "")
        monkeypatch.setenv("GPA_LLM_CHANNEL", "remote")
        from src.core.config import get_settings

        get_settings.cache_clear()
        state = {"query": "补贴 ROI 怎么样", "budget": 50000}
        update = await agent_plan_node(state)  # type: ignore[arg-type]
        assert update["plan"]["source"] == "rule"
        assert "plan:rule" in update["degradations"]
        assert update["selected_experts"] == ["subsidy"]
        assert all(sg["status"] == "pending" for sg in update["subgoals"])


class TestHeuristicBrain:
    def test_pending_subgoals_become_tool_calls(self):
        state = {
            "query": "q",
            "subgoals": [
                {"expert": "subsidy", "question": "补贴", "budget": 1000, "status": "pending"},
                {"expert": "ad", "question": "广告", "status": "done"},  # skipped
            ],
        }
        msg = _heuristic_brain_message(state)  # type: ignore[arg-type]
        assert [tc["name"] for tc in msg.tool_calls] == ["run_subsidy_analysis"]
        assert msg.tool_calls[0]["args"]["budget"] == 1000

    def test_no_pending_gives_final_placeholder(self):
        state = {"query": "q", "subgoals": [{"expert": "ad", "question": "x", "status": "done"}]}
        msg = _heuristic_brain_message(state)  # type: ignore[arg-type]
        assert not msg.tool_calls
        assert msg.content
