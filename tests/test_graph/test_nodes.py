"""Tests for src.graph.nodes - graph node functions with mocked LLM calls."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.graph.state import AnalysisStatus


# ---------------------------------------------------------------------------
# plan_node
# ---------------------------------------------------------------------------


class TestPlanNode:
    """Tests for the plan_node function."""

    @pytest.fixture
    def base_state(self):
        return {
            "query": "帮我分析最近的用户获取和转化数据",
            "scope": "full",
            "budget": 5000.0,
            "org_id": "org-1",
            "user_id": "u-1",
            "status": AnalysisStatus.PENDING,
            "expert_results": [],
            "execution_errors": [],
        }

    @pytest.mark.asyncio
    async def test_returns_selected_experts_and_plan(self, base_state):
        """plan_node selects experts and returns a plan dict."""
        from src.graph.nodes import plan_node

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "reasoning": "User asks about acquisition and conversion",
            "experts": ["prospect", "conversion"],
            "context_summary": "Growth analysis request",
        })

        with patch("src.guardrails.input_guard.validate_input") as mock_validate, \
             patch("src.core.llm_factory.create_llm") as mock_create_llm:
            mock_validate.return_value = MagicMock(passed=True, sanitized_input=base_state["query"])
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_create_llm.return_value = mock_llm

            result = await plan_node(base_state)

        assert "plan" in result
        assert "selected_experts" in result
        assert "prospect" in result["selected_experts"]
        assert "conversion" in result["selected_experts"]
        assert result["status"] == AnalysisStatus.EXECUTING

    @pytest.mark.asyncio
    async def test_rejects_invalid_input(self, base_state):
        """plan_node returns FAILED when input guardrail rejects."""
        from src.graph.nodes import plan_node

        base_state["query"] = "ab"

        with patch("src.guardrails.input_guard.validate_input") as mock_validate:
            mock_validate.return_value = MagicMock(
                passed=False, reason="Query too short"
            )
            result = await plan_node(base_state)

        assert result["status"] == AnalysisStatus.FAILED
        assert result["selected_experts"] == []
        assert "rejected" in result["plan"]["reasoning"].lower()

    @pytest.mark.asyncio
    async def test_keyword_fallback_on_llm_failure(self, base_state):
        """plan_node falls back to keyword matching when LLM fails."""
        from src.graph.nodes import plan_node

        with patch("src.guardrails.input_guard.validate_input") as mock_validate, \
             patch("src.core.llm_factory.create_llm") as mock_create_llm:
            mock_validate.return_value = MagicMock(passed=True, sanitized_input=base_state["query"])
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))
            mock_create_llm.return_value = mock_llm

            result = await plan_node(base_state)

        assert result["status"] == AnalysisStatus.EXECUTING
        # "用户获取" and "转化" are in the query, so keyword fallback should find them
        assert len(result["selected_experts"]) > 0

    @pytest.mark.asyncio
    async def test_fallback_to_all_experts_when_no_keywords_match(self):
        """When no keywords match and LLM fails, all experts are selected."""
        from src.graph.nodes import plan_node

        state = {
            "query": "general business review with no specific keywords",
            "scope": "",
            "budget": 1000.0,
            "org_id": "org-1",
            "user_id": "u-1",
            "status": AnalysisStatus.PENDING,
            "expert_results": [],
            "execution_errors": [],
        }

        with patch("src.guardrails.input_guard.validate_input") as mock_validate, \
             patch("src.core.llm_factory.create_llm") as mock_create_llm:
            mock_validate.return_value = MagicMock(passed=True, sanitized_input=state["query"])
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))
            mock_create_llm.return_value = mock_llm

            result = await plan_node(state)

        assert len(result["selected_experts"]) == 5


# ---------------------------------------------------------------------------
# execute_node
# ---------------------------------------------------------------------------


class TestExecuteNode:
    """Tests for the execute_node function."""

    @pytest.fixture
    def executing_state(self):
        return {
            "query": "analyze growth",
            "scope": "full",
            "budget": 5000.0,
            "org_id": "org-1",
            "user_id": "u-1",
            "status": AnalysisStatus.EXECUTING,
            "selected_experts": ["prospect", "conversion"],
            "expert_results": [],
            "execution_errors": [],
        }

    @pytest.mark.asyncio
    async def test_parallel_execution_with_mocked_experts(self, executing_state):
        """execute_node runs experts in parallel and collects results."""
        from src.graph.nodes import execute_node

        # Each call must return a fresh dict to avoid shared-mutation bugs
        def make_expert():
            inst = MagicMock()
            inst.analyze = AsyncMock(return_value={
                "success": True,
                "analysis": {"summary": "Analysis result"},
            })
            return inst

        with patch("src.graph.nodes._create_expert", side_effect=lambda key: make_expert()):
            result = await execute_node(executing_state)

        assert result["status"] == AnalysisStatus.EVALUATING
        assert len(result["expert_results"]) == 2
        # Both should have expert key set
        experts_in_result = {r["expert"] for r in result["expert_results"]}
        assert experts_in_result == {"prospect", "conversion"}

    @pytest.mark.asyncio
    async def test_handles_expert_exceptions(self, executing_state):
        """execute_node records errors when an expert throws."""
        from src.graph.nodes import execute_node

        # First expert succeeds, second throws
        good_expert = MagicMock()
        good_expert.analyze = AsyncMock(return_value={"success": True, "analysis": {}})

        bad_expert = MagicMock()
        bad_expert.analyze = AsyncMock(side_effect=RuntimeError("API error"))

        def mock_create(key):
            return good_expert if key == "prospect" else bad_expert

        with patch("src.graph.nodes._create_expert", side_effect=mock_create):
            result = await execute_node(executing_state)

        assert len(result["execution_errors"]) == 1
        assert "conversion" in result["execution_errors"][0]
        assert len(result["expert_results"]) == 2
        # Check the failed result
        failed_result = [r for r in result["expert_results"] if not r.get("success", True)]
        assert len(failed_result) == 1
        assert failed_result[0]["expert"] == "conversion"

    @pytest.mark.asyncio
    async def test_no_experts_selected(self):
        """execute_node handles empty expert list gracefully."""
        from src.graph.nodes import execute_node

        state = {
            "query": "test",
            "scope": "",
            "budget": 0,
            "selected_experts": [],
            "expert_results": [],
            "execution_errors": [],
        }
        result = await execute_node(state)
        assert result["status"] == AnalysisStatus.EVALUATING
        assert result["expert_results"] == []
        assert result["execution_errors"] == []


# ---------------------------------------------------------------------------
# evaluate_node
# ---------------------------------------------------------------------------


class TestEvaluateNode:
    """Tests for the evaluate_node function."""

    @pytest.fixture
    def evaluating_state(self):
        return {
            "query": "analyze growth",
            "scope": "full",
            "budget": 5000.0,
            "org_id": "org-1",
            "user_id": "u-1",
            "status": AnalysisStatus.EVALUATING,
            "selected_experts": ["prospect"],
            "expert_results": [
                {"expert": "prospect", "success": True, "analysis": {"summary": "result"}},
            ],
            "execution_errors": [],
            "refinement_round": 0,
        }

    @pytest.mark.asyncio
    async def test_returns_quality_scores(self, evaluating_state):
        """evaluate_node returns quality_scores dict."""
        from src.core.evaluator import QualityScore
        from src.graph.nodes import evaluate_node

        mock_scores = {
            "prospect": QualityScore(
                completeness=0.8, actionability=0.85, data_grounding=0.75, overall=0.8, reasoning="Good"
            ),
        }

        with patch("src.core.evaluator.batch_evaluate", new_callable=AsyncMock, return_value=mock_scores):
            result = await evaluate_node(evaluating_state)

        assert "quality_scores" in result
        assert "prospect" in result["quality_scores"]
        assert result["quality_scores"]["prospect"]["overall"] == 0.8

    @pytest.mark.asyncio
    async def test_sets_needs_refinement_below_threshold(self, evaluating_state):
        """needs_refinement is True when any score is below threshold (0.7)."""
        from src.core.evaluator import QualityScore
        from src.graph.nodes import evaluate_node

        mock_scores = {
            "prospect": QualityScore(
                completeness=0.4, actionability=0.5, data_grounding=0.3, overall=0.4, reasoning="Weak"
            ),
        }

        with patch("src.core.evaluator.batch_evaluate", new_callable=AsyncMock, return_value=mock_scores):
            result = await evaluate_node(evaluating_state)

        assert result["needs_refinement"] is True
        assert result["refinement_round"] == 1

    @pytest.mark.asyncio
    async def test_no_refinement_above_threshold(self, evaluating_state):
        """needs_refinement is False when all scores are above threshold."""
        from src.core.evaluator import QualityScore
        from src.graph.nodes import evaluate_node

        mock_scores = {
            "prospect": QualityScore(
                completeness=0.9, actionability=0.85, data_grounding=0.88, overall=0.88, reasoning="Great"
            ),
        }

        with patch("src.core.evaluator.batch_evaluate", new_callable=AsyncMock, return_value=mock_scores):
            result = await evaluate_node(evaluating_state)

        assert result["needs_refinement"] is False
        assert result["refinement_round"] == 0

    @pytest.mark.asyncio
    async def test_empty_results_skips_to_reporting(self):
        """When no expert results, evaluation skips directly to REPORTING."""
        from src.graph.nodes import evaluate_node

        state = {
            "query": "test",
            "expert_results": [],
            "execution_errors": [],
            "refinement_round": 0,
        }
        result = await evaluate_node(state)
        assert result["quality_scores"] == {}
        assert result["needs_refinement"] is False
        assert result["status"] == AnalysisStatus.REPORTING


# ---------------------------------------------------------------------------
# approval_node
# ---------------------------------------------------------------------------


class TestApprovalNode:
    """Tests for the approval_node function."""

    @pytest.mark.asyncio
    async def test_auto_approves_low_budget(self):
        """approval_node auto-approves when budget is below threshold."""
        from src.graph.nodes import approval_node

        state = {
            "budget": 5000.0,
            "approval_required": True,
        }
        result = await approval_node(state)
        assert result["approved"] is True
        assert result["status"] == AnalysisStatus.REPORTING

    @pytest.mark.asyncio
    async def test_requires_approval_high_budget(self):
        """approval_node requires approval when budget >= threshold."""
        from src.graph.nodes import approval_node

        state = {
            "budget": 15_000.0,
            "approval_required": True,
        }
        result = await approval_node(state)
        assert result["approved"] is None
        assert result["status"] == AnalysisStatus.AWAITING_APPROVAL

    @pytest.mark.asyncio
    async def test_auto_approves_when_not_required(self):
        """approval_node auto-approves when approval_required is False."""
        from src.graph.nodes import approval_node

        state = {
            "budget": 50_000.0,
            "approval_required": False,
        }
        result = await approval_node(state)
        assert result["approved"] is True
        assert result["status"] == AnalysisStatus.REPORTING

    @pytest.mark.asyncio
    async def test_auto_approves_zero_budget(self):
        """approval_node auto-approves when budget is 0 or not set."""
        from src.graph.nodes import approval_node

        state = {"budget": 0, "approval_required": True}
        result = await approval_node(state)
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_auto_approves_missing_budget(self):
        """approval_node auto-approves when budget is absent."""
        from src.graph.nodes import approval_node

        state = {"approval_required": True}
        result = await approval_node(state)
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_exact_threshold(self):
        """Budget exactly at threshold requires approval."""
        from src.graph.nodes import approval_node, BUDGET_APPROVAL_THRESHOLD

        state = {
            "budget": BUDGET_APPROVAL_THRESHOLD,
            "approval_required": True,
        }
        result = await approval_node(state)
        assert result["approved"] is None
        assert result["status"] == AnalysisStatus.AWAITING_APPROVAL


# ---------------------------------------------------------------------------
# report_node
# ---------------------------------------------------------------------------


class TestReportNode:
    """Tests for the report_node function."""

    @pytest.fixture
    def pre_report_state(self):
        return {
            "query": "分析增长数据",
            "scope": "full",
            "budget": 5000.0,
            "org_id": "org-1",
            "user_id": "u-1",
            "status": AnalysisStatus.REPORTING,
            "selected_experts": ["prospect", "conversion"],
            "expert_results": [
                {"expert": "prospect", "success": True, "analysis": {"summary": "User acquisition up 15%"}},
                {"expert": "conversion", "success": True, "analysis": {"summary": "Funnel conversion at 3.2%"}},
            ],
            "execution_errors": [],
            "quality_scores": {
                "prospect": {"overall": 0.85},
                "conversion": {"overall": 0.9},
            },
        }

    @pytest.mark.asyncio
    async def test_returns_final_report(self, pre_report_state):
        """report_node returns final_report, token_usage, cost_usd."""
        from src.graph.nodes import report_node

        mock_response = MagicMock()
        mock_response.content = "# Analysis Report\n\n## Executive Summary\nGrowth is strong."

        with patch("src.core.llm_factory.create_llm") as mock_create_llm, \
             patch("src.guardrails.output_guard.validate_output") as mock_validate:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_create_llm.return_value = mock_llm
            mock_validate.return_value = MagicMock(passed=True)

            result = await report_node(pre_report_state)

        assert "final_report" in result
        assert result["status"] == AnalysisStatus.COMPLETED
        assert isinstance(result["token_usage"], dict)
        assert "total_tokens" in result["token_usage"]
        assert isinstance(result["cost_usd"], float)

    @pytest.mark.asyncio
    async def test_handles_failed_experts_in_report(self, pre_report_state):
        """report_node handles experts that failed gracefully."""
        from src.graph.nodes import report_node

        pre_report_state["expert_results"] = [
            {"expert": "prospect", "success": True, "analysis": {"summary": "Good results"}},
            {"expert": "conversion", "success": False, "error": "API timeout"},
        ]

        mock_response = MagicMock()
        mock_response.content = "# Report with failures noted"

        with patch("src.core.llm_factory.create_llm") as mock_create_llm, \
             patch("src.guardrails.output_guard.validate_output") as mock_validate:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_create_llm.return_value = mock_llm
            mock_validate.return_value = MagicMock(passed=True)

            result = await report_node(pre_report_state)

        assert result["status"] == AnalysisStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fallback_report_on_llm_failure(self, pre_report_state):
        """report_node generates a fallback report when LLM fails."""
        from src.graph.nodes import report_node

        with patch("src.core.llm_factory.create_llm") as mock_create_llm, \
             patch("src.guardrails.output_guard.validate_output") as mock_validate:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))
            mock_create_llm.return_value = mock_llm
            mock_validate.return_value = MagicMock(passed=True)

            result = await report_node(pre_report_state)

        assert result["status"] == AnalysisStatus.COMPLETED
        assert "# Analysis Report" in result["final_report"]
        assert result["cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_output_guardrail_filters_report(self, pre_report_state):
        """report_node applies output guardrail and filters when needed."""
        from src.graph.nodes import report_node

        mock_response = MagicMock()
        mock_response.content = "password=secret123 some report text that is long enough"

        with patch("src.core.llm_factory.create_llm") as mock_create_llm, \
             patch("src.guardrails.output_guard.validate_output") as mock_validate:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_create_llm.return_value = mock_llm
            mock_validate.return_value = MagicMock(
                passed=False, reason="Sensitive information detected"
            )

            result = await report_node(pre_report_state)

        assert result["status"] == AnalysisStatus.COMPLETED
        assert "filtered" in result["final_report"].lower()
