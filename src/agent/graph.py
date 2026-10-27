"""Agent-led StateGraph: the Chief Agent tool-calling loop.

Topology::

    guard → agent ⇄ tools → evaluate_gate ─┬→ agent   (feedback, bounded rounds)
                                          └→ approval → report → END

- ``agent`` calls the LLM with bound tools and decides the next action.
- ``tools`` executes tool calls (experts / memory search), feeding digests
  back as ToolMessages.
- ``evaluate_gate`` scores the final answer; feedback re-enters the loop.
- ``approval`` interrupts (checkpointer + interrupt_before) for
  human-in-the-loop on high-budget runs.
- ``report`` emits the guardrailed final answer and writes memory.

The loop is bounded by ``Settings.max_refinement_rounds``; each agent/tools
step is bounded by the expert timeout, and the overall run by the caller.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.chief import (
    _route_after_approval,
    _route_after_gate,
    agent_approval_node,
    agent_plan_node,
    agent_report_node,
    agent_tools_node,
    chief_agent_node,
    evaluate_gate_node,
    should_continue,
)
from src.agent.state import AgentState
from src.graph.gate_node import kpi_gate_node
from src.graph.state import AnalysisStatus

logger = logging.getLogger(__name__)


async def guard_node(state: AgentState) -> dict:
    """Input guardrail before the agent loop starts (mirrors plan_node)."""
    from src.guardrails.input_guard import validate_input

    query = state.get("query", "")
    guard = validate_input(query, state.get("budget"))
    if guard.passed:
        return {"status": AnalysisStatus.EXECUTING}
    logger.warning("[guard] Input rejected: %s", guard.reason)
    return {
        "selected_experts": [],
        "status": AnalysisStatus.FAILED,
    }


def _route_after_guard(state: AgentState) -> str:
    if state.get("status") == AnalysisStatus.FAILED:
        return END
    return "plan"


def build_agent_graph() -> StateGraph:
    """Build (uncompiled) the agent-led StateGraph.

    Topology::

        guard → plan → agent ⇄ tools → evaluate ─┬→ agent (feedback, bounded)
                                                └→ approval → report → kpi_gate → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("guard", guard_node)
    graph.add_node("plan", agent_plan_node)
    graph.add_node("agent", chief_agent_node)
    graph.add_node("tools", agent_tools_node)
    graph.add_node("evaluate", evaluate_gate_node)
    graph.add_node("approval", agent_approval_node)
    graph.add_node("report", agent_report_node)
    graph.add_node("kpi_gate", kpi_gate_node)

    graph.set_entry_point("guard")
    graph.add_conditional_edges("guard", _route_after_guard, {"plan": "plan", END: END})
    graph.add_edge("plan", "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "evaluate": "evaluate"}
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "evaluate",
        _route_after_gate,
        {"agent": "agent", "approval": "approval"},
    )
    graph.add_conditional_edges(
        "approval",
        _route_after_approval,
        {"report": "report", END: END},
    )
    graph.add_edge("report", "kpi_gate")
    graph.add_edge("kpi_gate", END)

    logger.info("[build_agent_graph] agent-led StateGraph wired")
    return graph


async def build_compiled_agent_graph():
    """Compile the agent graph with checkpointing and approval interrupt."""
    from src.graph.checkpoint import get_checkpointer

    graph = build_agent_graph()
    checkpointer = await get_checkpointer()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["approval"])


async def run_agent_analysis(
    *,
    query: str,
    org_id: str = "default",
    user_id: str = "anonymous",
    scope: str | None = None,
    budget: float | None = None,
    data_path: str | None = None,
    approval_required: bool = False,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run a complete agent-led analysis; returns the final AgentState dict."""
    # Run-scoped cost attribution (concurrent runs no longer share a
    # process-global tracker).
    import uuid as _uuid

    from src.core.observability import trace_id_var
    from src.core.token_tracker import cost_scope

    if not thread_id:
        # Unique per run: a shared default thread leaked checkpointed state
        # across "independent" runs (kpi_snapshot from a previous run bled
        # into the next render). HITL callers pass an explicit stable id.
        thread_id = _uuid.uuid4().hex[:12]
    trace_id_var.set(f"{org_id}:{thread_id}")

    initial_state: AgentState = {  # type: ignore[typeddict-item]
        "query": query,
        "org_id": org_id,
        "user_id": user_id,
        "scope": scope,
        "budget": budget,
        "data_path": data_path,
        "approval_required": approval_required,
        "expert_results": [],
        "execution_errors": [],
        "messages": [],
        "status": AnalysisStatus.PENDING,
    }

    config = {"configurable": {"thread_id": f"{org_id}:{thread_id}"}}
    try:
        async with cost_scope():
            from src.graph.runtime import get_compiled_graph

            compiled = await get_compiled_graph("agent")
            result = await compiled.ainvoke(initial_state, config=config)
    except Exception as exc:
        # In agent-led mode the LLM *is* the orchestrator — when it is not
        # configured the heuristic brain (v9) still runs through the same
        # graph; a hard failure here means something else broke. Fail
        # cleanly with an actionable message instead of a stack trace.
        logger.error("[run_agent_analysis] agent loop failed: %s", exc)
        return {
            **initial_state,
            "status": AnalysisStatus.FAILED,
            "execution_errors": [f"agent loop: {exc}"],
            "final_report": (
                f"# Agent 运行失败\n\n## 错误\n{exc}\n\n"
                "请检查 GPA_LLM_BASE_URL / GPA_LLM_API_KEY 配置，或使用 "
                "`--mode workflow`（关键词路由，不依赖 LLM 决策）作为降级方案。"
            ),
        }

    # Guard-rejected inputs short-circuit before any message exists.
    if result.get("status") == AnalysisStatus.FAILED:
        result["final_report"] = f"输入被护栏拒绝：{query[:100]}"
    return result
