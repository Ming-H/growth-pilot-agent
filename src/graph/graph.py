"""LangGraph StateGraph DAG builder for GrowthPilot.

Builds the conditional DAG that orchestrates:
plan -> execute -> evaluate -> [refine -> execute | approval -> report | report]

The compiled graph can be invoked with an AnalysisState dict and will run
through the full analysis lifecycle with optional quality refinement loops
and human approval checkpoints.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from src.graph.state import AnalysisState, AnalysisStatus
from src.graph.nodes import (
    plan_node,
    execute_node,
    evaluate_node,
    refine_node,
    approval_node,
    report_node,
)

logger = logging.getLogger(__name__)

# Maximum refinement rounds before forcing progression
MAX_REFINEMENT_ROUNDS: int = 2


# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------

def _route_after_evaluate(state: AnalysisState) -> str:
    """Determine the next node after evaluation.

    Logic:
    1. If needs_refinement and refinement_round < MAX -> "execute" (re-run)
    2. Elif approval_required -> "approval"
    3. Else -> "report"
    """
    needs_refinement = state.get("needs_refinement", False)
    refinement_round = state.get("refinement_round", 0) or 0
    approval_required = state.get("approval_required", False)

    if needs_refinement and refinement_round < MAX_REFINEMENT_ROUNDS:
        logger.info(
            "[router] Routing to execute for refinement round %d",
            refinement_round,
        )
        return "execute"

    if approval_required:
        logger.info("[router] Routing to approval checkpoint")
        return "approval"

    logger.info("[router] Routing to report generation")
    return "report"


def _route_after_approval(state: AnalysisState) -> str:
    """Determine the next node after the approval checkpoint.

    If approved -> "report"
    If rejected -> END
    """
    approved = state.get("approved")

    if approved is True:
        logger.info("[router] Approved, routing to report")
        return "report"

    logger.info("[router] Not approved (or pending), routing to END")
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and return the compiled StateGraph DAG.

    Graph structure::

        plan -> execute -> evaluate -+-> execute (refinement loop)
                                      +-> approval -> report -> END
                                      +-> report -> END
                                      +-> END (rejected)

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(AnalysisState)

    # Add all nodes
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("refine", refine_node)
    graph.add_node("approval", approval_node)
    graph.add_node("report", report_node)

    # Set entry point
    graph.set_entry_point("plan")

    # Static edges
    graph.add_edge("plan", "execute")

    # After execute, always go to evaluate
    graph.add_edge("execute", "evaluate")

    # Conditional edge after evaluate
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "execute": "execute",   # re-run with refined set
            "approval": "approval",
            "report": "report",
        },
    )

    # Conditional edge after approval
    graph.add_conditional_edges(
        "approval",
        _route_after_approval,
        {
            "report": "report",
            END: END,
        },
    )

    # Report always goes to END
    graph.add_edge("report", END)

    # Compile and return
    compiled = graph.compile()
    logger.info("[build_graph] StateGraph DAG compiled successfully")
    return compiled


# ---------------------------------------------------------------------------
# Convenience: run a full analysis through the graph
# ---------------------------------------------------------------------------

async def run_analysis(
    *,
    query: str,
    org_id: str = "default",
    user_id: str = "anonymous",
    scope: str | None = None,
    budget: float | None = None,
    approval_required: bool = False,
) -> dict[str, Any]:
    """Run a complete analysis through the StateGraph DAG.

    This is the primary entry point for the graph-based analysis pipeline.

    Args:
        query: The user's growth analytics question.
        org_id: Organization identifier.
        user_id: User identifier.
        scope: Optional scope hint (e.g. "prospect", "full").
        budget: Optional budget for the analysis.
        approval_required: Whether human approval is required before reporting.

    Returns:
        Final AnalysisState dict after the graph completes.
    """
    compiled_graph = build_graph()

    initial_state: AnalysisState = {
        "query": query,
        "org_id": org_id,
        "user_id": user_id,
        "scope": scope,
        "budget": budget,
        "approval_required": approval_required,
        "expert_results": [],
        "execution_errors": [],
        "status": AnalysisStatus.PENDING,
    }

    result = await compiled_graph.ainvoke(initial_state)
    return result
