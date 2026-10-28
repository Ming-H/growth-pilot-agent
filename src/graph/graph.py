"""LangGraph StateGraph DAG builder for GrowthPilot.

Builds the conditional DAG that orchestrates:
plan -> execute -> evaluate -> [refine -> evaluate | approval -> report | report]

The compiled graph can be invoked with an AnalysisState dict and will run
through the full analysis lifecycle with optional quality refinement loops
and human approval checkpoints.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.graph.gate_node import kpi_gate_node
from src.graph.nodes import (
    approval_node,
    evaluate_node,
    execute_node,
    plan_node,
    refine_node,
    report_node,
)
from src.graph.state import AnalysisState, AnalysisStatus

logger = logging.getLogger(__name__)

# Note: max_refinement_rounds is configured in src.core.config.Settings


# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------


def _route_after_evaluate(state: AnalysisState) -> str:
    """Determine the next node after evaluation.

    Logic:
    1. If needs_refinement and refinement_round < MAX -> "refine" (re-evaluate)
    2. Elif approval_required -> "approval"
    3. Else -> "report"
    """
    needs_refinement = state.get("needs_refinement", False)
    refinement_round = state.get("refinement_round", 0) or 0
    approval_required = state.get("approval_required", False)

    from src.core.config import get_settings

    settings = get_settings()
    if needs_refinement and refinement_round < settings.max_refinement_rounds:
        logger.info(
            "[router] Routing to refine for refinement round %d",
            refinement_round,
        )
        return "refine"

    if approval_required:
        logger.info("[router] Routing to approval checkpoint")
        return "approval"

    logger.info("[router] Routing to report generation")
    return "report"


def _route_after_plan(state: AnalysisState) -> str:
    """Route after planning: execute if experts selected, else end.

    If the plan node selected experts for execution, route to the execute
    node.  If no experts were selected (e.g. the query was out of scope or
    the plan decided no analysis is needed), route directly to END.
    """
    selected = state.get("selected_experts", [])
    if selected:
        return "execute"
    logger.info("[router] No experts selected, routing to END")
    return END


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
    """Build and return the *uncompiled* StateGraph DAG.

    Graph structure::

        plan -> execute -> evaluate -+-> execute (refinement loop)
                                      +-> approval -> report -> END
                                      +-> report -> END
                                      +-> END (rejected)

    Returns:
        An uncompiled ``StateGraph``.  Call ``build_compiled_graph()`` to get
        a compiled graph with checkpointing and interrupt support, or call
        ``graph.compile()`` yourself for a bare compilation.
    """
    graph = StateGraph(AnalysisState)

    # Add all nodes
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("refine", refine_node)
    graph.add_node("approval", approval_node)
    graph.add_node("report", report_node)
    graph.add_node("kpi_gate", kpi_gate_node)

    # Set entry point
    graph.set_entry_point("plan")

    # Static edges
    # Conditional edge after plan: execute only if experts were selected
    graph.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "execute": "execute",
            END: END,  # no experts selected, skip to end
        },
    )

    # After execute, always go to evaluate
    graph.add_edge("execute", "evaluate")

    # Conditional edge after evaluate
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "refine": "refine",  # re-run only experts below quality threshold
            "approval": "approval",
            "report": "report",
        },
    )

    # After refine, always re-evaluate
    graph.add_edge("refine", "evaluate")

    # Conditional edge after approval
    graph.add_conditional_edges(
        "approval",
        _route_after_approval,
        {
            "report": "report",
            END: END,
        },
    )

    # Report passes the reconciliation gate before leaving the graph
    graph.add_edge("report", "kpi_gate")
    graph.add_edge("kpi_gate", END)

    logger.info("[build_graph] StateGraph DAG wired successfully (not yet compiled)")
    return graph


async def build_compiled_graph():
    """Build, compile and return the StateGraph DAG with checkpointing.

    This is the preferred entry point for running analyses.  It:

    1. Calls :func:`build_graph` to create the raw ``StateGraph``.
    2. Obtains a checkpointer via :func:`get_checkpointer`.
    3. Compiles the graph with ``checkpointer`` and
       ``interrupt_before=["approval"]`` so that the execution pauses at the
       approval node, enabling a human-in-the-loop workflow.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for invocation.
    """
    from src.graph.checkpoint import get_checkpointer

    graph = build_graph()
    checkpointer = await get_checkpointer()

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["approval"],
    )
    logger.info("[build_compiled_graph] Graph compiled with checkpointing and approval interrupt")
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
    data_path: str | None = None,
    approval_required: bool = False,
    thread_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run a complete analysis through the StateGraph DAG.

    This is the primary entry point for the graph-based analysis pipeline.
    The graph is compiled with checkpointing so that execution can pause at
    the ``approval`` node and resume later via ``thread_id``.

    Args:
        query: The user's growth analytics question.
        org_id: Organization identifier.
        user_id: User identifier.
        scope: Optional scope hint (e.g. "prospect", "full").
        budget: Optional budget for the analysis.
        data_path: Optional path to a data file (CSV/Parquet) consumed by
            experts that support external data (prospect, retention).
        approval_required: Whether human approval is required before reporting.
        thread_id: Unique identifier for the checkpoint thread.  Use the same
            ``thread_id`` when resuming after a human-approval interrupt.

    Returns:
        Final AnalysisState dict after the graph completes.
    """
    # Run-scoped cost attribution: every LLM call inside this context
    # (including worker threads) records to this run's tracker.
    import uuid as _uuid

    from src.core.observability import trace_id_var
    from src.core.token_tracker import cost_scope

    if not thread_id:
        # Unique per run: a shared default thread leaked checkpointed state
        # across "independent" runs (kpi_snapshot from a previous run bled
        # into the next render). HITL callers pass an explicit stable id.
        thread_id = _uuid.uuid4().hex[:12]
    trace_id_var.set(f"{org_id}:{thread_id}")

    async with cost_scope():
        from src.graph.runtime import get_compiled_graph

        compiled_graph = await get_compiled_graph("workflow")

        initial_state: AnalysisState = {
            "query": query,
            "org_id": org_id,
            "user_id": user_id,
            "session_id": session_id,
            "scope": scope,
            "budget": budget,
            "data_path": data_path,
            "approval_required": approval_required,
            "expert_results": [],
            "execution_errors": [],
            "status": AnalysisStatus.PENDING,
        }

        config = {"configurable": {"thread_id": f"{org_id}:{thread_id}"}}
        result = await compiled_graph.ainvoke(initial_state, config=config)
    return result
