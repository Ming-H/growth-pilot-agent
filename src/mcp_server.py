"""MCP server: deterministic growth-analysis tools over the six-expert tool layer.

Protocol: Model Context Protocol (stdio), Python SDK v2 (``MCPServer``).
Start with ``gpa mcp``; register with hosts such as Claude Code via
``claude mcp add growth-pilot -- <cmd running `gpa mcp`>``.

Design contract (docs/plans/2026-11-09-mcp-and-task-benchmark-design.md):

- Tools take **high-level business knobs** (budget, top_k), never
  dataframes. Data comes from the seeded synthetic generators (seed=42)
  shared with the experts' demo paths; every result carries
  ``data_source: "synthetic"`` — the honesty marker used elsewhere in
  the platform (e.g. the ad expert).
- **Zero LLM, zero memory writes, zero gate enforcement here.** This
  layer exposes raw deterministic tool output; narrative reports with
  the KPI reconciliation gate come from ``gpa analyze``.
- Determinism: same arguments ⇒ same result (regression-tested).

Tool descriptions follow the platform's ACI discipline
(``src/agent/tools.py``): purpose / when-to-call / args / failure-modes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)

server = MCPServer(
    name="growth-pilot",
    title="GrowthPilot growth tools",
    description=(
        "Deterministic user-growth analysis tools: subsidy optimisation, "
        "LTV ranking, churn watchlist, funnel bottleneck, cross-channel "
        "budget allocation."
    ),
    instructions=(
        "All tools run on synthetic demo data (seed=42, data_source=synthetic) "
        "and are deterministic: same arguments, same result. They compute "
        "facts only — no LLM, no narrative. For gated narrative reports use "
        "`gpa analyze`."
    ),
)


def _tool(fn):  # register while keeping the function directly callable/testable
    server.tool()(fn)
    return fn


# ---------------------------------------------------------------------------
# Subsidy expert (causal ATE → integer-programming coupon allocation)
# ---------------------------------------------------------------------------


@_tool
def subsidy_optimize(
    total_budget_yuan: float,
    min_coupon_yuan: float = 5,
    max_coupon_yuan: float = 50,
) -> dict[str, Any]:
    """Allocate a coupon budget across user segments to maximise incremental orders.

    Purpose: integer-programming coupon allocation over segment-level causal
    effects; ATE estimated on the demo subsidy experiment (backdoor adjustment),
    using the exact data recipe of the subsidy expert pipeline.

    When to call: "how should I split this coupon budget", per-segment coupon
    sizing, subsidy ROI questions — e.g. "50000 yuan coupon budget, how to
    allocate across segments".

    Args:
        total_budget_yuan: total subsidy budget in CNY (must be > 0).
        min_coupon_yuan: minimum per-user coupon in CNY (default 5).
        max_coupon_yuan: maximum per-user coupon in CNY (default 50).

    Failure modes: {"error": ...} when budget <= 0 or no valid coupon amounts;
    falls back to a greedy solver (solver: "greedy") when PuLP is unavailable.

    Data: synthetic demo (seed=42) — data_source: synthetic.
    """
    from src.tools.subsidy.budget_optimizer import BudgetOptimizer
    from src.tools.subsidy.causal_engine import CausalInferenceEngine

    if total_budget_yuan <= 0:
        return {"error": "total_budget_yuan must be positive"}
    if max_coupon_yuan < min_coupon_yuan:
        return {"error": "max_coupon_yuan must be >= min_coupon_yuan"}

    # Same data path as the subsidy expert (load_sample_data recipe) so MCP
    # numbers agree with `gpa analyze` runs and the task-benchmark oracle.
    from src.agents.subsidy import SubsidyExpert

    data = SubsidyExpert._load_or_generate_data("")
    if "treatment" not in data.columns:
        if "group" in data.columns:
            data["treatment"] = (data["group"] != "control").astype(int)
        elif "subsidy_amount" in data.columns:
            data["treatment"] = (data["subsidy_amount"] > 0).astype(int)
    treatment_col = "treatment" if "treatment" in data.columns else "subsidy_amount"
    outcome_col = "converted" if "converted" in data.columns else "revenue"
    confounders = [c for c in ("age", "city_tier", "historical_orders") if c in data.columns]

    try:
        ate_result = CausalInferenceEngine().estimate_ate(
            data=data,
            treatment=treatment_col,
            outcome=outcome_col,
            confounders=confounders,
            method="backdoor",
        )
    except Exception as exc:  # noqa: BLE001 — honest degrade, same as the expert
        logger.warning("ATE estimation failed in MCP tool: %s", exc)
        ate_result = {"ate": 0.12, "method": "heuristic"}
    if not isinstance(ate_result, dict) or "error" in ate_result:
        ate_result = {"ate": 0.12, "method": "heuristic"}

    if "segment" in data.columns:
        segments = {str(k): int(v) for k, v in data["segment"].value_counts().to_dict().items()}
    else:
        segments = {"new_user": 500, "active": 300, "dormant": 200}
    ate_val = float(ate_result.get("ate", 0.12))
    causal_effects = {
        seg: {"ate": ate_val, "base_conversion_rate": 0.12, "coupon_amount_used": 10.0}
        for seg in segments
    }

    result = BudgetOptimizer().optimize_allocation(
        user_segments=segments,
        causal_effects=causal_effects,
        total_budget=float(total_budget_yuan),
        min_coupon=float(min_coupon_yuan),
        max_coupon=float(max_coupon_yuan),
        coupon_step=5,
    )
    if not isinstance(result, dict) or "error" in result:
        err = (result or {}).get("error", "optimiser failed")
        return {"error": err, "data_source": "synthetic"}

    used = float(result.get("total_budget_used", 0) or 0)
    incremental = float(result.get("expected_incremental_orders", 0) or 0)
    result["roi"] = round(incremental / used, 4) if used > 0 else None
    result["ate"] = round(ate_val, 4)
    result["ate_significant_at_05"] = bool(ate_result.get("significant_at_05", False))
    result["data_source"] = "synthetic"
    return result


# ---------------------------------------------------------------------------
# Prospect expert (feature engineering → LTV prediction)
# ---------------------------------------------------------------------------


@_tool
def ltv_rank(top_k: int = 5) -> dict[str, Any]:
    """Rank demo users by predicted lifetime value (highest first).

    Purpose: LightGBM LTV regression over behavioural features (same feature
    matrix and synthetic-target recipe as the prospect expert), returning the
    top-K users and, when segment metadata is available, mean LTV per segment.

    When to call: "which users have the highest LTV", user-value ranking,
    seed-list prioritisation questions.

    Args:
        top_k: number of top users to return (1-100, default 5).

    Failure modes: {"error": ...} when the feature matrix is empty or top_k
    is out of range; LTV targets are synthetic (heavy-tailed, feature-driven).

    Data: synthetic demo (seed=42) — data_source: synthetic.
    """
    from src.agents.prospect import ProspectExpert
    from src.tools.prospect.feature_engine import FeatureEngine
    from src.tools.prospect.ltv_predictor import LTVPredictor

    if not 1 <= top_k <= 100:
        return {"error": "top_k must be within [1, 100]"}

    raw = ProspectExpert._load_data("")
    features = FeatureEngine().build_feature_matrix(raw)
    if features is None or features.empty:
        return {"error": "feature matrix is empty"}

    # Synthetic LTV targets — verbatim recipe from the prospect expert so
    # numbers agree with the graph's demo path.
    rng = np.random.RandomState(42)
    numeric = features.select_dtypes(include=[np.number]).fillna(0)
    if numeric.empty:
        return {"error": "no numeric features available"}
    w = rng.normal(0, 1.0, size=numeric.shape[1])
    logits = numeric.values @ w
    z = (logits - logits.mean()) / (logits.std() + 1e-9)
    synthetic_ltv = np.exp(0.2 * z + rng.normal(0, 0.3, size=len(features))) * 200

    predictor = LTVPredictor()
    if predictor.train(features, synthetic_ltv) is None:
        return {"error": "LTV training failed"}
    preds = np.asarray(predictor.predict_ltv(features)).ravel()

    frame = pd.DataFrame({"predicted_ltv": preds}, index=features.index)
    frame = frame.sort_values("predicted_ltv", ascending=False)
    top_users = [
        {"user_id": str(uid), "predicted_ltv": round(float(val), 2)}
        for uid, val in frame.head(top_k)["predicted_ltv"].items()
    ]

    result: dict[str, Any] = {
        "top_users": top_users,
        "mean_predicted_ltv": round(float(frame["predicted_ltv"].mean()), 2),
        "n_users": int(len(frame)),
        "data_source": "synthetic",
    }

    # Segment-level aggregation when the demo profile carries a segment column.
    profile = raw.get("user_profile") if isinstance(raw, dict) else None
    if (
        isinstance(profile, pd.DataFrame)
        and "segment" in profile.columns
        and "user_id" in profile.columns
    ):
        seg_map = profile.set_index("user_id")["segment"]
        seg = frame.join(seg_map, how="left")
        grouped = seg.groupby("segment")["predicted_ltv"].mean().sort_values(ascending=False)
        result["segment_ltv"] = [
            {"segment": str(s), "mean_predicted_ltv": round(float(v), 2)}
            for s, v in grouped.items()
        ]

    return result


# ---------------------------------------------------------------------------
# Retention expert (churn model → watchlist)
# ---------------------------------------------------------------------------


@_tool
def churn_watchlist(top_k: int = 10) -> dict[str, Any]:
    """List the demo users at highest churn risk, with model quality metrics.

    Purpose: LightGBM churn classifier (class-imbalance aware) trained on the
    demo churn dataset; returns the top-K risk users plus AUC / average
    precision so the caller can judge model reliability.

    When to call: "who is about to churn", retention watchlist, win-back
    target selection questions.

    Args:
        top_k: number of users to return (1-100, default 10).

    Failure modes: {"error": ...} when training fails or top_k out of range;
    scores are probabilities in [0, 1] — higher means higher churn risk.

    Data: synthetic demo (seed=42) — data_source: synthetic.
    """
    from src.tools.retention.churn_predictor import ChurnPredictor

    if not 1 <= top_k <= 100:
        return {"error": "top_k must be within [1, 100]"}

    X, _y = ChurnPredictor.generate_sample_data(n_samples=2000)
    predictor = ChurnPredictor()
    metrics = predictor.train(X, _y)
    if not isinstance(metrics, dict):
        return {"error": "churn model training failed"}
    probs = np.asarray(predictor.predict_churn_risk(X)).ravel()

    order = np.argsort(-probs)[:top_k]
    watchlist = [
        {"user_id": f"U{int(i):05d}", "churn_risk": round(float(probs[i]), 4)} for i in order
    ]
    return {
        "auc": round(float(metrics.get("auc", 0.0)), 4),
        "average_precision": round(float(metrics.get("average_precision", 0.0)), 4),
        "watchlist": watchlist,
        "data_source": "synthetic",
    }


# ---------------------------------------------------------------------------
# Conversion expert (funnel bottleneck)
# ---------------------------------------------------------------------------


@_tool
def funnel_bottleneck() -> dict[str, Any]:
    """Locate the biggest drop-off stage in the demo conversion funnel.

    Purpose: per-stage conversion rates with bottleneck identification,
    diagnosis and suggested actions (same analyser as the conversion expert).

    When to call: "where do we lose users in the funnel", conversion
    bottleneck, step-wise drop-off questions.

    Args: none — runs on the demo funnel (1000 users, seed=42).

    Failure modes: {"error": ...} when the funnel data cannot be normalised.

    Data: synthetic demo (seed=42) — data_source: synthetic.
    """
    from src.tools.common.data_loader import DataLoader
    from src.tools.conversion.funnel_analyzer import FunnelAnalyzer

    counts = DataLoader().generate_funnel_data(n_users=1000)
    result = FunnelAnalyzer().analyze_funnel(counts)
    if not isinstance(result, dict) or "error" in result:
        return {"error": (result or {}).get("error", "funnel analysis failed")}
    # Normalise the bottleneck field so hosts always find one key.
    if "bottleneck_stage" not in result:
        bottleneck = result.get("bottleneck") or result.get("worst_step")
        if isinstance(bottleneck, dict):
            bottleneck = bottleneck.get("stage") or bottleneck.get("step")
        if bottleneck:
            result["bottleneck_stage"] = bottleneck
    result["data_source"] = "synthetic"
    return result


# ---------------------------------------------------------------------------
# Budget expert (cross-channel global allocation)
# ---------------------------------------------------------------------------


@_tool
def budget_allocate(total_budget_yuan: float) -> dict[str, Any]:
    """Allocate a marketing budget across channels via integer programming.

    Purpose: global cross-channel allocation that maximises total expected
    outcome under budget, using the same concave response curves and IP
    solver as the budget expert (no sibling-expert KPI calibration here —
    context_kpis defaults to neutral).

    When to call: "how to split budget across channels", marketing-mix
    allocation, channel budget planning questions.

    Args:
        total_budget_yuan: total marketing budget in CNY (must be > 0).

    Failure modes: {"error": ...} when budget <= 0 or the allocator fails;
    unallocated remainder is reported as "unallocated".

    Data: synthetic demo (seed=42) — data_source: synthetic.
    """
    from src.tools.budget import ChannelResponseModel, GlobalBudgetAllocator

    if total_budget_yuan <= 0:
        return {"error": "total_budget_yuan must be positive"}

    model = ChannelResponseModel(total_budget=float(total_budget_yuan), context_kpis={})
    grids = {c: model.grid(c) for c in model.channels}
    plan = GlobalBudgetAllocator().allocate(grids, total_budget=float(total_budget_yuan))
    if not isinstance(plan, dict) or "error" in plan:
        return {"error": (plan or {}).get("error", "allocator failed")}
    plan["data_source"] = "synthetic"
    return plan


# ---------------------------------------------------------------------------
# Entry point (`gpa mcp`)
# ---------------------------------------------------------------------------


def run_server() -> None:
    """Run the MCP server over stdio (blocking)."""
    import anyio

    anyio.run(server.run_stdio_async)
