"""Task-level benchmark — "τ-bench for Growth" (docs/plans/2026-11-09 design).

Each task is an end-to-end growth question whose ground truth is computed
INDEPENDENTLY of the graph, via the seeded direct-tool recipes the MCP
server exposes (``src.mcp_server``). This answers a different question
than the router/heuristic evals — not "did the pipeline run" but
"did the system deliver the right answer":

    route   — expected experts ⊆ actually-executed experts
    numeric — ground-truth value present in ``kpi_snapshot`` (tolerance)
    gate    — ``kpi_gate_result.ok`` is True (every report number traceable)
    honesty — (refusal tasks) no fabricated KPIs for out-of-domain queries:
              the snapshot must stay EMPTY when the query is not about growth

All runs use the zero-LLM workflow mode (deterministic, CI-safe). The
oracle layer re-runs the same seed-42 recipes on the same data, so a
numeric miss means the graph failed to deliver the tool's number —
never that the oracle drifted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Anchor:
    """A ground-truth numeric expectation against ``kpi_snapshot``."""

    label: str
    value: float
    tolerance: float = 0.002
    snapshot_key: str | None = None  # None → match against any snapshot value


@dataclass(frozen=True)
class GrowthTask:
    """One benchmark task: query + routing expectation + numeric anchors."""

    id: str
    query: str
    budget: float
    expected_experts: frozenset[str]
    anchors: tuple[Anchor, ...] = field(default_factory=tuple)
    kind: str = "analytical"  # analytical | honesty
    description: str = ""


@dataclass
class TaskScore:
    """Per-task scoring outcome (pure — computed from a run result dict)."""

    route_ok: bool
    numeric_ok: bool
    gate_ok: bool

    @property
    def ok(self) -> bool:
        return self.route_ok and self.numeric_ok and self.gate_ok

    def cells(self) -> dict[str, bool]:
        return {
            "route": self.route_ok,
            "numeric": self.numeric_ok,
            "gate": self.gate_ok,
        }


# ---------------------------------------------------------------------------
# Ground-truth oracles (independent of the graph; seed-42 direct recipes)
# ---------------------------------------------------------------------------


def gt_subsidy_ate(budget: float = 50000.0) -> float:
    """Subsidy ATE via the MCP direct recipe (causal engine on demo data)."""
    from src.mcp_server import subsidy_optimize

    result = subsidy_optimize(total_budget_yuan=budget)
    return float(result["ate"])


def gt_prospect_users() -> int:
    """Demo prospect population size via the MCP LTV recipe."""
    from src.mcp_server import ltv_rank

    return int(ltv_rank(top_k=1)["n_users"])


def gt_churn_auc() -> float:
    """Churn model AUC via the MCP watchlist recipe."""
    from src.mcp_server import churn_watchlist

    return float(churn_watchlist(top_k=5)["auc"])


def gt_funnel_min_step_rate() -> float:
    """Worst funnel step rate, recomputed from raw counts (fully independent

    of both the graph and the FunnelAnalyzer implementation)."""
    import numpy as np

    from src.tools.common.data_loader import DataLoader

    counts = DataLoader().generate_funnel_data(n_users=1000)
    values = [float(v) for v in counts.values()]
    if len(values) < 2 or any(v <= 0 for v in values[:-1]):
        return -1.0
    ratios = [values[i + 1] / values[i] for i in range(len(values) - 1)]
    return float(np.min(ratios))


def _alloc_sum(snapshot: dict[str, Any]) -> float:
    return sum(
        float(v)
        for k, v in snapshot.items()
        if k.startswith("alloc_") and isinstance(v, (int, float))
    )


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------


def builtin_tasks() -> list[GrowthTask]:
    """The built-in benchmark suite (7 tasks, design doc §A)."""
    subsidy_ate = gt_subsidy_ate(50000.0)
    churn_auc = gt_churn_auc()
    funnel_rate = gt_funnel_min_step_rate()
    prospect_n = gt_prospect_users()

    return [
        GrowthTask(
            id="subsidy_plan_50k",
            query="5万补贴预算怎么分配ROI最高",
            budget=50000.0,
            expected_experts=frozenset({"subsidy"}),
            anchors=(Anchor("subsidy_ate", subsidy_ate, 0.002, "subsidy_ate"),),
            description="预算分配问题须由补贴专家给出可追溯的因果ATE",
        ),
        GrowthTask(
            id="prospect_population",
            query="用户意向和LTV分析",
            budget=0.0,
            expected_experts=frozenset({"prospect"}),
            anchors=(Anchor("total_users", float(prospect_n), 0.5, "total_users"),),
            description="潜客分析覆盖的用户总量须与数据源一致",
        ),
        GrowthTask(
            id="churn_watchlist",
            query="流失风险最高的用户有哪些",
            budget=0.0,
            expected_experts=frozenset({"retention"}),
            anchors=(Anchor("churn_auc", churn_auc, 0.002, "churn_model_auc"),),
            description="流失预警的模型质量(AUC)须与直调真值一致",
        ),
        GrowthTask(
            id="funnel_bottleneck",
            query="转化漏斗哪里流失最严重",
            budget=0.0,
            expected_experts=frozenset({"conversion"}),
            anchors=(Anchor("min_step_rate", funnel_rate, 0.01, "funnel_bottleneck_rate"),),
            description="漏斗瓶颈比率须与原始计数直算一致",
        ),
        GrowthTask(
            id="budget_channels_500k",
            query="50万预算在渠道之间怎么分",
            budget=500000.0,
            expected_experts=frozenset({"budget"}),
            anchors=(
                Anchor("budget_total", 500000.0, 0.5, "budget_total"),
                Anchor(
                    "alloc_within_budget",
                    500000.0,  # matched as: allocation sum <= this value
                    0.0,
                    None,
                ),
            ),
            description="跨渠道分配必须花满上报预算且不超支",
        ),
        GrowthTask(
            id="multi_domain_subsidy_churn",
            query="补贴预算5万，同时看用户流失情况",
            budget=50000.0,
            expected_experts=frozenset({"subsidy", "retention"}),
            anchors=(Anchor("subsidy_ate", subsidy_ate, 0.002, "subsidy_ate"),),
            description="复合问题须同时路由两位专家且数字可追溯",
        ),
        GrowthTask(
            id="honesty_refusal",
            query="帮我写一首关于货运的短诗",
            budget=0.0,
            expected_experts=frozenset(),
            kind="honesty",
            description="非增长问题不得伪造KPI——宁拒不编(可评测约束)",
        ),
    ]


# ---------------------------------------------------------------------------
# Scoring (pure functions over run-result dicts)
# ---------------------------------------------------------------------------


def value_present(
    snapshot: dict[str, Any], target: float, tolerance: float, key: str | None
) -> bool:
    """True when *target* matches a snapshot value within *tolerance*.

    ``key`` pins the expectation to a specific snapshot field; ``None``
    matches against any numeric snapshot value.
    """
    if key is not None:
        v = snapshot.get(key)
        return isinstance(v, (int, float)) and abs(float(v) - target) <= tolerance
    return any(
        isinstance(v, (int, float)) and abs(float(v) - target) <= tolerance
        for v in snapshot.values()
    )


def allocation_within_budget(snapshot: dict[str, Any], budget: float) -> bool:
    """Sum of ``alloc_*`` snapshot values must be positive and <= budget."""
    total = _alloc_sum(snapshot)
    return 0 < total <= budget + 1e-6


def score_task(task: GrowthTask, result: dict[str, Any]) -> TaskScore:
    """Score one run result against a task (pure; see module docstring)."""
    experts = set((result.get("expert_results") or {}).keys())
    snapshot: dict[str, Any] = result.get("kpi_snapshot") or {}

    if task.kind == "honesty":
        # Honest refusal: no growth KPIs may be fabricated for an
        # out-of-domain query. Empty snapshot = pass, whatever the routing.
        return TaskScore(route_ok=True, numeric_ok=not snapshot, gate_ok=True)

    route_ok = task.expected_experts <= experts
    numeric_ok = all(
        value_present(snapshot, a.value, a.tolerance, a.snapshot_key) for a in task.anchors
    )
    if task.id == "budget_channels_500k":
        numeric_ok = numeric_ok and allocation_within_budget(snapshot, 500000.0)

    gate = result.get("kpi_gate_result") or {}
    # No claims in the report → vacuously traceable; otherwise the gate
    # must have run and passed (ok=True, no violations).
    gate_ok = bool(gate.get("ok", True)) and not (gate.get("violations") or [])
    return TaskScore(route_ok=route_ok, numeric_ok=numeric_ok, gate_ok=gate_ok)
