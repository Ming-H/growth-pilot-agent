"""GrowthPilot workflow — LangGraph StateGraph orchestration.

Routes all analysis through the LangGraph StateGraph DAG (src.graph.graph),
replacing the old Chief Agent ReAct loop.  The ``run_workflow()`` function
signature is preserved for backward compatibility with CLI and web API.

Design references:
- LangGraph StateGraph: declarative DAG with conditional edges
- OpenAI Runner: execution engine for the agent loop
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_success(result_state: dict[str, Any]) -> bool:
    """Determine if the analysis completed successfully."""
    status = result_state.get("status")
    if status is None:
        return False
    status_str = status.value if hasattr(status, "value") else str(status)
    return status_str != "failed"


# ---------------------------------------------------------------------------
# Convenience runner (backward-compatible interface)
# ---------------------------------------------------------------------------


async def run_workflow(
    *,
    query: str = "",
    data_path: str = "",
    budget: float = 0,
    scope: str = "",
) -> dict[str, Any]:
    """Run the GrowthPilot analysis via the LangGraph StateGraph DAG.

    This function preserves the old interface for CLI and web API compatibility.
    Internally it delegates to :func:`src.graph.graph.run_analysis`.

    Returns a dict with keys: success, query, scope, analysis_summary,
    strategy_recommendation, expert_results, errors, report, metadata.
    """
    from src.graph.graph import run_analysis

    result_state = await run_analysis(
        query=query,
        scope=scope or None,
        budget=budget or None,
    )

    # Map AnalysisState fields to the old result dict format for backward compat
    expert_results_list: list[dict[str, Any]] = result_state.get("expert_results", [])
    expert_results_map = {
        r.get("expert", f"unknown_{i}"): r
        for i, r in enumerate(expert_results_list)
    }
    final_report = result_state.get("final_report", "")
    execution_errors: list[str] = result_state.get("execution_errors", [])

    # Build KPI snapshot from expert results
    kpi_snapshot: dict[str, Any] = {}
    for r in expert_results_list:
        if r.get("expert") == "prospect":
            kpi_snapshot["total_users"] = r.get("user_count", 0)
            kpi_snapshot["intent_auc"] = r.get("intent_metrics", {}).get("auc", 0)
        elif r.get("expert") == "subsidy":
            kpi_snapshot["expected_roi"] = r.get("expected_roi", 0)
        elif r.get("expert") == "ad":
            kpi_snapshot["ad_cpa"] = r.get("expected_cpa", 0)

    result: dict[str, Any] = {
        "success": _is_success(result_state),
        "query": query,
        "scope": scope or "full",
        "analysis_summary": final_report,
        "strategy_recommendation": "",
        "expert_results": expert_results_map,
        "errors": execution_errors,
        "report": final_report,
        "metadata": {
            "token_usage": result_state.get("token_usage", {}),
            "cost_usd": result_state.get("cost_usd", 0),
        },
        "kpi_snapshot": kpi_snapshot,
    }

    return result
