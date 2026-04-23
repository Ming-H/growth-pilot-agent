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


class AnalysisState(TypedDict):
    """Graph state for the LangGraph StateGraph DAG.

    Fields are grouped by the phase that primarily reads/writes them.
    List fields use ``Annotated[..., operator.add]`` reducers so that
    parallel nodes (e.g. expert execution) can append independently and
    LangGraph merges the results automatically.
    """

    # ── Input ────────────────────────────────────────────────────────────
    query: str
    scope: NotRequired[str | None]
    budget: NotRequired[float | None]
    org_id: str
    user_id: str

    # ── Planning ─────────────────────────────────────────────────────────
    plan: NotRequired[dict[str, Any] | None]
    selected_experts: NotRequired[list[str]]

    # ── Execution ────────────────────────────────────────────────────────
    expert_results: Annotated[list[dict[str, Any]], operator.add]
    execution_errors: Annotated[list[str], operator.add]

    # ── Evaluation ───────────────────────────────────────────────────────
    quality_scores: NotRequired[dict[str, Any] | None]
    needs_refinement: NotRequired[bool]
    refinement_round: NotRequired[int]

    # ── Approval ─────────────────────────────────────────────────────────
    approval_required: NotRequired[bool]
    approved: NotRequired[bool | None]

    # ── Output ───────────────────────────────────────────────────────────
    final_report: NotRequired[str | None]
    status: AnalysisStatus
    token_usage: NotRequired[dict[str, Any]]
    cost_usd: NotRequired[float]
