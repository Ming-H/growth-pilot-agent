"""Tests for src.graph.state - AnalysisState and AnalysisStatus."""
from __future__ import annotations

import enum
import operator

import pytest

from src.graph.state import AnalysisState, AnalysisStatus


# ---------------------------------------------------------------------------
# AnalysisStatus enum
# ---------------------------------------------------------------------------


class TestAnalysisStatus:
    """Tests for the AnalysisStatus enum."""

    def test_has_all_eight_values(self):
        """AnalysisStatus defines exactly 8 enum members."""
        expected = {
            "PENDING",
            "PLANNING",
            "EXECUTING",
            "EVALUATING",
            "AWAITING_APPROVAL",
            "REPORTING",
            "COMPLETED",
            "FAILED",
        }
        actual = {m.name for m in AnalysisStatus}
        assert actual == expected

    def test_enum_values_are_strings(self):
        """Each enum value is a lowercase string."""
        for member in AnalysisStatus:
            assert isinstance(member.value, str)
            assert member.value == member.name.lower()

    def test_pending_value(self):
        assert AnalysisStatus.PENDING.value == "pending"

    def test_failed_value(self):
        assert AnalysisStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# AnalysisState TypedDict construction
# ---------------------------------------------------------------------------


class TestAnalysisStateConstruction:
    """Tests for building AnalysisState dicts with required/optional fields."""

    def test_required_fields_only(self):
        """AnalysisState can be built with just the required fields."""
        state: AnalysisState = {
            "query": "分析最近的增长数据",
            "org_id": "org-001",
            "user_id": "user-001",
            "status": AnalysisStatus.PENDING,
            "expert_results": [],
            "execution_errors": [],
        }
        assert state["query"] == "分析最近的增长数据"
        assert state["status"] == AnalysisStatus.PENDING
        assert state["expert_results"] == []
        assert state["execution_errors"] == []

    def test_with_optional_fields(self):
        """AnalysisState accepts optional fields when provided."""
        state: AnalysisState = {
            "query": "帮我优化转化率",
            "scope": "conversion",
            "budget": 5000.0,
            "org_id": "org-002",
            "user_id": "user-002",
            "status": AnalysisStatus.PLANNING,
            "plan": {"reasoning": "test", "experts": ["conversion"], "context_summary": "..."},
            "selected_experts": ["conversion"],
            "expert_results": [],
            "execution_errors": [],
            "quality_scores": {"conversion": {"overall": 0.85}},
            "needs_refinement": False,
            "refinement_round": 0,
            "approval_required": False,
            "approved": True,
            "final_report": "# Report",
            "token_usage": {"total_tokens": 100},
            "cost_usd": 0.002,
        }
        assert state["budget"] == 5000.0
        assert state["selected_experts"] == ["conversion"]
        assert state["quality_scores"]["conversion"]["overall"] == 0.85
        assert state["cost_usd"] == 0.002

    def test_optional_fields_default_absent(self):
        """Optional fields (NotRequired) are not present when not provided."""
        state: AnalysisState = {
            "query": "query",
            "org_id": "org-1",
            "user_id": "u-1",
            "status": AnalysisStatus.PENDING,
            "expert_results": [],
            "execution_errors": [],
        }
        # Optional fields should not be present
        assert "scope" not in state
        assert "budget" not in state
        assert "plan" not in state
        assert "final_report" not in state

    def test_optional_fields_can_be_none(self):
        """Optional fields can be explicitly set to None."""
        state: AnalysisState = {
            "query": "q",
            "org_id": "org-1",
            "user_id": "u-1",
            "status": AnalysisStatus.PENDING,
            "scope": None,
            "budget": None,
            "plan": None,
            "expert_results": [],
            "execution_errors": [],
        }
        assert state["scope"] is None
        assert state["budget"] is None
        assert state["plan"] is None


# ---------------------------------------------------------------------------
# Annotated reducer (operator.add) behaviour
# ---------------------------------------------------------------------------


class TestAnnotatedReducers:
    """Tests that the operator.add annotation works for list fields."""

    def test_expert_results_reducer_adds(self):
        """Simulate LangGraph merging expert_results from parallel branches."""
        base: list[dict] = [{"expert": "prospect", "score": 0.9}]
        incoming: list[dict] = [{"expert": "conversion", "score": 0.8}]
        merged = operator.add(base, incoming)
        assert len(merged) == 2
        assert merged[0]["expert"] == "prospect"
        assert merged[1]["expert"] == "conversion"

    def test_execution_errors_reducer_adds(self):
        """Simulate LangGraph merging execution_errors from parallel branches."""
        base: list[str] = ["prospect: timeout"]
        incoming: list[str] = ["retention: connection error"]
        merged = operator.add(base, incoming)
        assert len(merged) == 2
        assert "prospect: timeout" in merged
        assert "retention: connection error" in merged

    def test_expert_results_reducer_empty_plus_results(self):
        """Adding results to an empty list works."""
        base: list[dict] = []
        incoming: list[dict] = [{"expert": "ad", "success": True}]
        merged = operator.add(base, incoming)
        assert len(merged) == 1

    def test_expert_results_reducer_results_plus_empty(self):
        """Adding an empty list preserves existing results."""
        base: list[dict] = [{"expert": "ad", "success": True}]
        incoming: list[dict] = []
        merged = operator.add(base, incoming)
        assert len(merged) == 1

    def test_multiple_parallel_branches_merge(self):
        """Three parallel branches all merge correctly."""
        results_a: list[dict] = [{"expert": "prospect"}]
        results_b: list[dict] = [{"expert": "conversion"}]
        results_c: list[dict] = [{"expert": "subsidy"}]
        merged = operator.add(operator.add(results_a, results_b), results_c)
        assert len(merged) == 3
        experts = {r["expert"] for r in merged}
        assert experts == {"prospect", "conversion", "subsidy"}

    def test_annotation_type_is_operator_add(self):
        """Verify that the Annotated type hint uses operator.add."""
        import typing

        hints = typing.get_type_hints(AnalysisState, include_extras=True)
        # expert_results should be Annotated[list[...], operator.add]
        er_hint = hints.get("expert_results")
        assert er_hint is not None
        # Annotated type: args are (list[..., ], operator.add)
        args = typing.get_args(er_hint)
        assert len(args) == 2
        assert args[1] is operator.add
