"""Tests for src.graph.state - AnalysisState and AnalysisStatus."""
from __future__ import annotations

import enum
import operator

import pytest

from src.graph.state import AnalysisState, AnalysisStatus, _replace_by_expert


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
# Annotated reducer (_replace_by_expert for expert_results, operator.add for errors)
# ---------------------------------------------------------------------------


class TestExpertResultsReducer:
    """Tests for the _replace_by_expert reducer used by expert_results."""

    def test_new_expert_appended(self):
        """A new expert's result is appended to existing results."""
        base = [{"expert": "prospect", "score": 0.9}]
        incoming = [{"expert": "conversion", "score": 0.8}]
        merged = _replace_by_expert(base, incoming)
        assert len(merged) == 2
        experts = {r["expert"] for r in merged}
        assert experts == {"prospect", "conversion"}

    def test_existing_expert_replaced(self):
        """A re-run expert replaces its old result instead of duplicating."""
        base = [{"expert": "prospect", "score": 0.5}]
        incoming = [{"expert": "prospect", "score": 0.9}]
        merged = _replace_by_expert(base, incoming)
        assert len(merged) == 1
        assert merged[0]["score"] == 0.9

    def test_empty_incoming_preserves_existing(self):
        """An empty incoming list returns existing results unchanged."""
        base = [{"expert": "prospect", "score": 0.9}]
        merged = _replace_by_expert(base, [])
        assert len(merged) == 1

    def test_empty_base_accepts_new(self):
        """An empty base accepts new results."""
        merged = _replace_by_expert([], [{"expert": "ad", "success": True}])
        assert len(merged) == 1

    def test_mixed_replace_and_append(self):
        """One replaced, one new, one untouched."""
        base = [
            {"expert": "prospect", "score": 0.5},
            {"expert": "conversion", "score": 0.8},
        ]
        incoming = [
            {"expert": "prospect", "score": 0.9},
            {"expert": "retention", "score": 0.7},
        ]
        merged = _replace_by_expert(base, incoming)
        assert len(merged) == 3
        by_expert = {r["expert"]: r for r in merged}
        assert by_expert["prospect"]["score"] == 0.9  # replaced
        assert by_expert["conversion"]["score"] == 0.8  # untouched
        assert by_expert["retention"]["score"] == 0.7  # new


class TestExecutionErrorsReducer:
    """Tests that execution_errors still uses operator.add (always appends)."""

    def test_errors_append(self):
        base = ["prospect: timeout"]
        incoming = ["retention: connection error"]
        merged = operator.add(base, incoming)
        assert len(merged) == 2

    def test_annotation_type_expert_results(self):
        """Verify that expert_results uses _replace_by_expert reducer."""
        import typing
        hints = typing.get_type_hints(AnalysisState, include_extras=True)
        er_hint = hints.get("expert_results")
        assert er_hint is not None
        args = typing.get_args(er_hint)
        assert len(args) == 2
        assert args[1] is _replace_by_expert

    def test_annotation_type_execution_errors(self):
        """Verify that execution_errors still uses operator.add."""
        import typing
        hints = typing.get_type_hints(AnalysisState, include_extras=True)
        er_hint = hints.get("execution_errors")
        assert er_hint is not None
        args = typing.get_args(er_hint)
        assert len(args) == 2
        assert args[1] is operator.add
