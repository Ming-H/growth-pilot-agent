"""Expert Agent sub-graphs for LangGraph StateGraph.

Each Expert Agent is wrapped as a LangGraph sub-graph with two nodes:
    execute -> validate

This provides:
- Independent testability per expert
- Internal state management
- Full BaseAgent.run() lifecycle (including hooks)
- Pydantic validation at sub-graph boundary

Design reference:
- LangGraph Sub-graph: encapsulate complex agent logic
- OpenAI Agents SDK: structured output at every agent boundary
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.core.models import EXPERT_RESULT_MODELS as _RESULT_MODELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-graph state
# ---------------------------------------------------------------------------

class ExpertSubState(TypedDict):
    """Internal state for an expert sub-graph execution."""
    expert_key: str
    params: dict[str, Any]
    errors: list[str]
    validated_result: dict[str, Any]


# ---------------------------------------------------------------------------
# Sub-graph node implementations
# ---------------------------------------------------------------------------

async def execute_node(state: ExpertSubState) -> dict:
    """Execute expert via full BaseAgent lifecycle."""
    from src.graph.nodes import _create_expert

    expert_key = state["expert_key"]
    params = state["params"]
    errors: list[str] = list(state.get("errors", []))

    expert = _create_expert(expert_key)
    result = await expert.analyze(params)
    return {"validated_result": result, "errors": errors}


async def validate_node(state: ExpertSubState) -> dict:
    """Validate expert output against Pydantic model."""
    expert_key = state["expert_key"]
    pipeline_results = dict(state.get("validated_result", {}))
    errors: list[str] = list(state.get("errors", []))

    if errors:
        pipeline_results["errors"] = errors
    pipeline_results["expert"] = expert_key

    # Pydantic validation
    model_cls = _RESULT_MODELS.get(expert_key)
    if model_cls:
        try:
            validated = model_cls(**pipeline_results)
            validated_dict = validated.model_dump(exclude_none=True)
            validated_dict["expert"] = expert_key
            validated_dict["_validated"] = True
            pipeline_results = validated_dict
        except Exception as exc:
            logger.warning(
                "[subgraph:%s] Pydantic validation failed: %s",
                expert_key, exc,
            )

    return {
        "validated_result": pipeline_results,
    }


# ---------------------------------------------------------------------------
# Sub-graph builder
# ---------------------------------------------------------------------------

def build_expert_subgraph(expert_key: str) -> StateGraph:
    """Build a sub-graph for a specific expert agent.

    Args:
        expert_key: One of "prospect", "conversion", "subsidy", "retention", "ad".

    Returns:
        An uncompiled StateGraph for the expert.
    """
    graph = StateGraph(ExpertSubState)

    graph.add_node("execute", execute_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("execute")
    graph.add_edge("execute", "validate")
    graph.add_edge("validate", END)

    logger.debug("[subgraph] Built sub-graph for expert: %s", expert_key)
    return graph


def compile_expert_subgraph(expert_key: str):
    """Build and compile a sub-graph for a specific expert agent."""
    graph = build_expert_subgraph(expert_key)
    return graph.compile()


async def run_expert_subgraph(expert_key: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run an expert as a compiled sub-graph.

    Args:
        expert_key: Expert identifier.
        params: Parameters for the expert pipeline.

    Returns:
        Validated expert result dict.
    """
    compiled = compile_expert_subgraph(expert_key)

    initial_state: ExpertSubState = {
        "expert_key": expert_key,
        "params": params,
        "errors": [],
        "validated_result": {},
    }

    result = await compiled.ainvoke(initial_state)
    return result.get("validated_result", {})
