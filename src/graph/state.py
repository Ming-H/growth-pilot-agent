"""LangGraph AnalysisState schema for the StateGraph DAG orchestration.

Replaces the flat AgentState (src/core/state.py) with a richer schema that
tracks the full analysis lifecycle: planning, parallel execution, quality
evaluation, optional human approval, and final reporting.

The state uses Annotated reducers for list fields so that parallel node
outputs are automatically merged by LangGraph.
"""

from __future__ import annotations

import enum
import operator
from typing import Annotated, Any, NotRequired, TypedDict


class AnalysisStatus(enum.Enum):
    """Lifecycle status for an analysis run."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    AWAITING_APPROVAL = "awaiting_approval"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


def _replace_by_expert(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reducer that replaces results by expert key instead of appending.

    When a refinement loop re-runs an expert, the new result replaces the
    old one for the same expert key, preventing duplicate accumulation.
    Results without an ``expert`` key are appended as before.
    """
    if not new:
        return existing
    replace_keys = {item.get("expert") for item in new if "expert" in item}
    kept = [r for r in existing if r.get("expert") not in replace_keys]
    return kept + new


def _merge_by_key(
    existing: list[dict[str, Any]],
    new: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    """Merge list-of-dicts by a key field (upsert semantics)."""
    out = {item.get(key, i): item for i, item in enumerate(existing)}
    for i, item in enumerate(new):
        out[item.get(key, f"_{i}")] = item
    return list(out.values())


class AnalysisState(TypedDict):
    """Graph state for the LangGraph StateGraph DAG.

    Fields are grouped by the phase that primarily reads/writes them.
    ``expert_results`` uses a custom reducer that merges by expert key,
    preventing duplicate accumulation during refinement loops.
    """

    # ── Input ────────────────────────────────────────────────────────────
    query: str
    scope: NotRequired[str | None]
    budget: NotRequired[float | None]
    data_path: NotRequired[str | None]
    org_id: str
    user_id: str
    session_id: NotRequired[str | None]  # L0 chat session (memory layer)

    # ── Planning ─────────────────────────────────────────────────────────
    plan: NotRequired[dict[str, Any] | None]
    selected_experts: NotRequired[list[str]]
    # v9 planning pillar: structured subgoals.
    # Each: {"expert": str, "question": str, "budget": float|None,
    #        "status": "pending"|"done"|"failed"}
    subgoals: Annotated[list[dict[str, Any]], lambda a, b: _merge_by_key(a, b, "expert")]

    # ── Execution ────────────────────────────────────────────────────────
    expert_results: Annotated[list[dict[str, Any]], _replace_by_expert]
    execution_errors: Annotated[list[str], operator.add]

    # ── Evaluation / 对账闸门 ─────────────────────────────────────────────
    quality_scores: NotRequired[dict[str, Any] | None]
    needs_refinement: NotRequired[bool]
    refinement_round: NotRequired[int]
    kpi_snapshot: NotRequired[dict[str, Any]]
    kpi_gate_result: NotRequired[dict[str, Any] | None]

    # ── Approval ─────────────────────────────────────────────────────────
    approval_required: NotRequired[bool]
    approved: NotRequired[bool | None]

    # ── Output / observability ───────────────────────────────────────────
    final_report: NotRequired[str | None]
    status: AnalysisStatus
    token_usage: NotRequired[dict[str, Any]]
    cost_usd: NotRequired[float]
    # Visible degradation markers (tally discipline): plan:rule, eval:rule,
    # report:template, kpi_gate:degraded, kpi_gate:annotated, agent_loop:capped
    degradations: Annotated[list[str], operator.add]
    # Per-node trace entries {"node", "dur_ms", "detail"} (diagnostic tier)
    trace: Annotated[list[dict[str, Any]], operator.add]
    # Agent-loop step quadruples {"thought","action","obs","gate"}
    agent_steps: Annotated[list[dict[str, Any]], operator.add]
