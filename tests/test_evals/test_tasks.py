"""Tests for the task-level benchmark (τ-bench for Growth) and the bugs it caught.

Regression guards:
- scoring helpers (pure)
- router honesty guard (out-of-domain query must route to nothing)
- report generator must report the TRUE churn count, never len(top-5 display
  sample) — the B7 sibling caught by the `churn_watchlist` gate violation
- end-to-end: subsidy task + honesty task through the real graph (zero-LLM)
"""

from __future__ import annotations

import pytest

from src.evals.tasks import (
    Anchor,
    GrowthTask,
    allocation_within_budget,
    score_task,
    value_present,
)
from src.graph.nodes import _intent_classify, _looks_growth_related
from src.report.generator import ReportGenerator

# ---------------------------------------------------------------------------
# Pure scoring helpers
# ---------------------------------------------------------------------------


class TestValuePresent:
    def test_pinned_key_match_within_tolerance(self) -> None:
        assert value_present({"ate": 0.0346}, 0.0345, 0.002, "ate")

    def test_pinned_key_miss(self) -> None:
        assert not value_present({"ate": 0.05}, 0.03, 0.002, "ate")

    def test_any_value_match(self) -> None:
        assert value_present({"a": 1.0, "b": 42.0}, 42.0, 0.0, None)

    def test_ignores_non_numeric(self) -> None:
        assert not value_present({"a": "42"}, 42.0, 0.0, None)


class TestAllocationWithinBudget:
    def test_within_budget(self) -> None:
        assert allocation_within_budget({"alloc_ad": 150.0, "alloc_subsidy": 210.0}, 500.0)

    def test_over_budget_fails(self) -> None:
        assert not allocation_within_budget({"alloc_ad": 600.0}, 500.0)

    def test_empty_allocation_fails(self) -> None:
        assert not allocation_within_budget({}, 500.0)


def _analytical_task() -> GrowthTask:
    return GrowthTask(
        id="t",
        query="q",
        budget=1.0,
        expected_experts=frozenset({"subsidy"}),
        anchors=(Anchor("ate", 0.5, 0.01, "subsidy_ate"),),
    )


class TestScoreTask:
    def test_all_ok(self) -> None:
        result = {
            "expert_results": {"subsidy": {}},
            "kpi_snapshot": {"subsidy_ate": 0.505},
            "kpi_gate_result": {"ok": True, "violations": []},
        }
        assert score_task(_analytical_task(), result).ok

    def test_route_miss(self) -> None:
        result = {
            "expert_results": {"retention": {}},
            "kpi_snapshot": {"subsidy_ate": 0.5},
            "kpi_gate_result": {"ok": True},
        }
        assert not score_task(_analytical_task(), result).route_ok

    def test_numeric_miss(self) -> None:
        result = {
            "expert_results": {"subsidy": {}},
            "kpi_snapshot": {"subsidy_ate": 0.9},
            "kpi_gate_result": {"ok": True},
        }
        assert not score_task(_analytical_task(), result).numeric_ok

    def test_gate_violation_fails(self) -> None:
        result = {
            "expert_results": {"subsidy": {}},
            "kpi_snapshot": {"subsidy_ate": 0.5},
            "kpi_gate_result": {"ok": False, "violations": [{"raw": "5"}]},
        }
        assert not score_task(_analytical_task(), result).gate_ok


class TestHonestyScoring:
    def _honesty_task(self) -> GrowthTask:
        return GrowthTask(
            id="h",
            query="q",
            budget=0.0,
            expected_experts=frozenset(),
            kind="honesty",
        )

    def test_empty_snapshot_passes(self) -> None:
        assert score_task(self._honesty_task(), {"kpi_snapshot": {}}).ok

    def test_fabricated_snapshot_fails(self) -> None:
        result = {"expert_results": {"subsidy": {}}, "kpi_snapshot": {"roi": 1.2}}
        assert not score_task(self._honesty_task(), result).ok


# ---------------------------------------------------------------------------
# Router honesty guard
# ---------------------------------------------------------------------------


class TestRouterHonesty:
    def test_looks_growth_related(self) -> None:
        assert _looks_growth_related("分析一下本月货运节奏")
        assert _looks_growth_related("用户增长情况如何")
        assert not _looks_growth_related("帮我写一首关于货运的短诗")
        assert not _looks_growth_related("今天天气怎么样")

    def test_out_of_domain_routes_to_nothing(self) -> None:
        assert _intent_classify("帮我写一首关于货运的短诗") == []

    def test_growth_vocabulary_fallback_still_works(self) -> None:
        # Unexpected wording that hits no expert keyword must still fall
        # back to all experts when growth vocabulary is present.
        selected = _intent_classify("做个用户层面的整体分析")
        assert len(selected) > 0


# ---------------------------------------------------------------------------
# Report generator: TRUE churn count (B7 sibling regression)
# ---------------------------------------------------------------------------


class TestReportChurnCount:
    def test_reports_true_count_not_display_sample_length(self) -> None:
        # Report state format: expert_results is a LIST of per-expert dicts
        # carrying an "expert" key (see ReportGenerator._get_expert_result).
        state = {
            "expert_results": [
                {
                    "expert": "retention",
                    "high_risk_users": [{"user_id": f"U{i}"} for i in range(5)],
                    "churn_factors": ["feature_0"],
                    "kpis": {"churn_high_risk_count": 492.0},
                }
            ]
        }
        section = ReportGenerator()._retention_section(state)
        assert "高风险用户数: 492.0" in section
        assert "高风险用户数: 5" not in section


# ---------------------------------------------------------------------------
# End-to-end through the real graph (zero-LLM, deterministic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subsidy_task_end_to_end() -> None:
    from src.evals.tasks import builtin_tasks, score_task
    from src.graph import nodes as graph_nodes
    from src.graph.workflow import run_workflow

    task = next(t for t in builtin_tasks() if t.id == "subsidy_plan_50k")
    graph_nodes._clear_expert_cache()
    result = await run_workflow(query=task.query, scope="", budget=task.budget, mode="workflow")
    score = score_task(task, result)
    assert score.ok, f"cells={score.cells()}"


@pytest.mark.asyncio
async def test_honesty_task_end_to_end() -> None:
    from src.evals.tasks import builtin_tasks, score_task
    from src.graph import nodes as graph_nodes
    from src.graph.workflow import run_workflow

    task = next(t for t in builtin_tasks() if t.id == "honesty_refusal")
    graph_nodes._clear_expert_cache()
    result = await run_workflow(query=task.query, scope="", budget=task.budget, mode="workflow")
    score = score_task(task, result)
    assert score.ok, f"cells={score.cells()}"
    assert not (result.get("kpi_snapshot") or {}), "no KPIs may be fabricated"
