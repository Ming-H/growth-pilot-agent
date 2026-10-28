"""GrowthPilot orchestration — agent-led by default, fixed DAG as fallback.

``run_workflow()`` preserves the historical interface for the CLI and web
API and dispatches on ``Settings.orchestration_mode``:

- ``"agent"`` (default): Chief Agent tool-calling loop
  (:func:`src.agent.graph.run_agent_analysis`) — the LLM dynamically decides
  which experts/tools to invoke, in which order, and when to answer.
- ``"workflow"``: fixed ``plan → execute → evaluate → refine/approval/report``
  DAG (:func:`src.graph.graph.run_analysis`) — deterministic routing.

Both modes return the same result contract (see ``run_workflow``).
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


def _build_compat_result(
    result_state: dict[str, Any],
    *,
    query: str,
    scope: str,
) -> dict[str, Any]:
    """Map a final graph state (either mode) to the legacy result contract."""
    expert_results_list: list[dict[str, Any]] = result_state.get("expert_results", [])
    expert_results_map = {
        r.get("expert", f"unknown_{i}"): r for i, r in enumerate(expert_results_list)
    }
    final_report = result_state.get("final_report", "")
    execution_errors: list[str] = result_state.get("execution_errors", [])

    # Merge each expert's standardised `kpis` dict into the snapshot
    kpi_snapshot: dict[str, Any] = {}
    for r in expert_results_list:
        expert_kpis = r.get("kpis", {})
        if isinstance(expert_kpis, dict):
            kpi_snapshot.update(expert_kpis)

    return {
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
        # v9 observability pass-through (CLI -v / --json)
        "degradations": result_state.get("degradations", []),
        "trace": result_state.get("trace", []),
        "agent_steps": result_state.get("agent_steps", []),
        "kpi_gate_result": result_state.get("kpi_gate_result"),
    }


# ---------------------------------------------------------------------------
# Convenience runner (backward-compatible interface)
# ---------------------------------------------------------------------------


async def run_workflow(
    *,
    query: str = "",
    data_path: str = "",
    budget: float = 0,
    scope: str = "",
    mode: str = "",
    org_id: str = "default",
    user_id: str = "anonymous",
    approval_required: bool = False,
    thread_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run a GrowthPilot analysis in the configured orchestration mode.

    Args:
        query: The user's growth analytics question.
        data_path: Optional path to a data file (CSV/Parquet).
        budget: Optional budget for the analysis.
        scope: Optional scope hint.
        mode: Force ``"agent"`` or ``"workflow"``; empty string uses
            ``Settings.orchestration_mode`` (default: agent).
        org_id: Organisation — scopes memory writes and HITL threads (A1).
        user_id: Initiating user.
        approval_required: Request human approval before reporting.
        thread_id: Stable checkpoint thread (HITL resume); unique per run
            when omitted.
        session_id: L0 chat session for memory continuity.

    Returns a dict with keys: success, query, scope, analysis_summary,
    strategy_recommendation, expert_results, errors, report, metadata,
    kpi_snapshot.
    """
    from src.core.config import get_settings

    settings = get_settings()
    resolved_mode = mode or settings.orchestration_mode

    common = dict(
        query=query,
        scope=scope or None,
        budget=budget or None,
        data_path=data_path or None,
        org_id=org_id,
        user_id=user_id,
        approval_required=approval_required,
        thread_id=thread_id,  # type: ignore[arg-type]
        session_id=session_id,
    )

    if resolved_mode == "agent":
        from src.agent.graph import run_agent_analysis

        logger.info("[run_workflow] mode=agent (Chief Agent tool-calling loop)")
        result_state = await run_agent_analysis(**common)
    else:
        from src.graph.graph import run_analysis

        logger.info("[run_workflow] mode=workflow (fixed StateGraph DAG)")
        result_state = await run_analysis(**common)

    return _build_compat_result(result_state, query=query, scope=scope)
