"""Tests for src.evals.evaluator - AgentEvaluator and EvalResult."""

from __future__ import annotations

import json

import pytest

from src.evals.evaluator import (
    AgentEvaluator,
    EvalDimensionScore,
    EvalResult,
)


class TestEvalResult:
    """Tests for the EvalResult model."""

    def test_eval_result_creation(self):
        r = EvalResult(agent_name="prospect", input_query="test query")
        assert r.agent_name == "prospect"
        assert r.input_query == "test query"
        assert r.dimensions == {}

    def test_overall_score_empty(self):
        r = EvalResult()
        assert r.overall_score == 0.0

    def test_overall_score_with_dimensions(self):
        r = EvalResult(
            dimensions={
                "task_completion": EvalDimensionScore(score=0.8),
                "output_quality": EvalDimensionScore(score=0.9),
            },
        )
        # task_completion weight=0.35, output_quality weight=0.35
        # total_weight = 0.35 + 0.35 = 0.7
        # weighted_sum = 0.8*0.35 + 0.9*0.35 = 0.28 + 0.315 = 0.595
        expected = 0.595 / 0.7
        assert abs(r.overall_score - expected) < 0.01

    def test_dimension_accessors(self):
        r = EvalResult(
            dimensions={
                "task_completion": EvalDimensionScore(score=0.7),
                "tool_efficiency": EvalDimensionScore(score=0.6),
                "output_quality": EvalDimensionScore(score=0.8),
                "latency": EvalDimensionScore(score=0.9),
            },
        )
        assert r.task_completion == 0.7
        assert r.tool_efficiency == 0.6
        assert r.output_quality == 0.8
        assert r.latency == 0.9

    def test_dimension_accessors_default(self):
        r = EvalResult()
        # When no dimensions exist, .get returns None; properties handle it
        assert r.task_completion == 0.0
        assert r.latency == 0.0


class TestEvalDimensionScore:
    """Tests for EvalDimensionScore model."""

    def test_valid_score(self):
        s = EvalDimensionScore(score=0.75, reason="Good")
        assert s.score == 0.75
        assert s.reason == "Good"

    def test_score_bounds(self):
        with pytest.raises(Exception):
            EvalDimensionScore(score=1.5)
        with pytest.raises(Exception):
            EvalDimensionScore(score=-0.1)

    def test_score_at_boundaries(self):
        EvalDimensionScore(score=0.0)
        EvalDimensionScore(score=1.0)


class TestTaskCompletionScoring:
    """Tests for _eval_task_completion."""

    def test_with_expected_keys_all_found(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_task_completion(
            output_str='{"user_count": 100, "analysis": "good"}',
            expected_output=None,
            expected_keys=["user_count", "analysis"],
        )
        assert score.score == 1.0
        assert len(score.details["found_keys"]) == 2

    def test_with_expected_keys_partial(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_task_completion(
            output_str='{"user_count": 100}',
            expected_output=None,
            expected_keys=["user_count", "analysis", "segment_summary"],
        )
        assert score.score == pytest.approx(1 / 3, abs=0.01)
        assert len(score.details["missing_keys"]) == 2

    def test_with_expected_output(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_task_completion(
            output_str="user growth rate is 15% and conversion is 3%",
            expected_output="user growth conversion",
            expected_keys=[],
        )
        assert score.score > 0.0

    def test_basic_check_short(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_task_completion(
            output_str="hi",
            expected_output=None,
            expected_keys=[],
        )
        assert score.score == 0.5

    def test_basic_check_long(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_task_completion(
            output_str="x" * 200,
            expected_output=None,
            expected_keys=[],
        )
        assert score.score == 1.0


class TestToolEfficiencyScoring:
    """Tests for _eval_tool_efficiency."""

    def test_optimal_calls(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        calls = [{"tool": f"tool_{i}"} for i in range(5)]
        score = ev._eval_tool_efficiency(calls, "prospect")
        assert score.score == 1.0

    def test_no_calls_orchestrator(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_tool_efficiency([], "orchestrator")
        assert score.score == 1.0

    def test_no_calls_agent(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_tool_efficiency([], "prospect")
        assert score.score == 0.1

    def test_too_many_calls(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        calls = [{"tool": f"tool_{i}"} for i in range(20)]
        score = ev._eval_tool_efficiency(calls, "prospect")
        assert score.score < 1.0


class TestLatencyScoring:
    """Tests for _eval_latency."""

    def test_excellent(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_latency(1.0)
        assert score.score == 1.0

    def test_good(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_latency(3.0)
        assert 0.8 <= score.score <= 1.0

    def test_acceptable(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_latency(7.0)
        assert 0.6 <= score.score <= 0.8

    def test_slow(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_latency(20.0)
        assert 0.1 <= score.score <= 0.6

    def test_unacceptable(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_latency(60.0)
        assert score.score == 0.1

    def test_none_latency(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        score = ev._eval_latency(None)
        assert score.score == 0.5


class TestParseJudgeResponse:
    """Tests for _parse_judge_response."""

    def test_valid_json(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        text = '{"overall_score": 8, "reason": "Good"}'
        result = ev._parse_judge_response(text)
        assert result["overall_score"] == 8

    def test_json_in_code_block(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        text = '```json\n{"overall_score": 7}\n```'
        result = ev._parse_judge_response(text)
        assert result["overall_score"] == 7

    def test_unparseable_returns_default(self):
        ev = AgentEvaluator(llm=None)  # type: ignore
        result = ev._parse_judge_response("not json at all")
        assert result["overall_score"] == 5.0
