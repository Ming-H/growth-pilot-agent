"""LangGraph node functions for the GrowthPilot StateGraph DAG.

Each node is a pure function that receives the current AnalysisState and
returns a partial state update dict.  The graph builder (src/graph/builder.py)
wires these nodes together with conditional edges.

Node contract:
    Input:  AnalysisState (full state, read-only)
    Output: dict with a subset of AnalysisState keys to update
"""
from __future__ import annotations

from src.graph.state import AnalysisState


async def plan_node(state: AnalysisState) -> dict:
    """Chief Agent planning: parse the user query and select expert agents.

    Reads ``query``, ``scope``, and ``budget`` from the state.  Uses an LLM
    to decompose the query into an ExecutionPlan and picks the experts that
    should be invoked.

    Returns:
        dict with keys ``plan``, ``selected_experts``, ``status``.
    """
    ...


async def execute_node(state: AnalysisState) -> dict:
    """Parallel expert execution.

    Reads ``plan`` and ``selected_experts`` to dispatch work to the relevant
    ExpertAgent subclasses.  Results are appended via the ``expert_results``
    reducer (operator.add) so that parallel branches merge automatically.

    Returns:
        dict with keys ``expert_results``, ``execution_errors``, ``status``.
    """
    ...


async def evaluate_node(state: AnalysisState) -> dict:
    """Quality evaluation of expert results.

    Reads ``expert_results`` and ``query`` and uses the Evaluator to score
    each expert's output on completeness, actionability, and data-grounding.

    Returns:
        dict with keys ``quality_scores``, ``needs_refinement``,
        ``refinement_round``, ``status``.
    """
    ...


async def refine_node(state: AnalysisState) -> dict:
    """Retry experts whose output fell below the quality threshold.

    Reads ``quality_scores`` and ``refinement_round``.  For each expert with
    an overall score below the threshold, re-invokes the expert with the
    evaluator's feedback appended as additional context.

    Returns:
        dict with keys ``expert_results``, ``execution_errors``,
        ``refinement_round``, ``needs_refinement``, ``status``.
    """
    ...


async def approval_node(state: AnalysisState) -> dict:
    """Human approval checkpoint.

    When ``approval_required`` is True, the graph pauses here (LangGraph
    interrupt) and waits for a human decision.  The resumed call sets
    ``approved`` to True or False.

    Returns:
        dict with keys ``approved``, ``status``.
    """
    ...


async def report_node(state: AnalysisState) -> dict:
    """Generate the final analysis report.

    Reads ``expert_results``, ``quality_scores``, and the accumulated context
    to synthesize a structured report with summary, strategy recommendations,
    and a KPI snapshot.

    Returns:
        dict with keys ``final_report``, ``token_usage``, ``cost_usd``,
        ``status``.
    """
    ...
