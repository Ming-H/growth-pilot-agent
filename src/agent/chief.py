"""Chief Agent nodes for the agent-led orchestration mode.

Implements the classic agent loop on top of LangGraph:

    agent ⇄ tools   (LLM decides which tools to call, in which order,
                     whether results suffice, and when to answer)

After the agent produces a final answer (no tool calls), a quality gate
scores it; below-threshold answers get the evaluator feedback injected back
into the conversation for a bounded number of self-correction rounds.
Human approval (budget-gated) interrupts before the report is emitted,
exactly as in workflow mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.state import AgentState
from src.graph.state import AnalysisStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CHIEF_SYSTEM_PROMPT = """\
你是 GrowthPilot 的首席增长 Agent（Chief Agent），负责直接指挥一组领域专家工具完成用户增长分析。

## 可用工具
{tool_descriptions}

## 工作规则
1. **先思考再调用**：分析用户问题，判断需要哪些专家视角；若问题可能与本组织
历史分析相关，先调用 search_memory。
2. **按需调用**：你可以多次调用工具、观察结果后决定下一步——这是你的核心职责。
不同专家可以并行覆盖（分别调用），也可以串联深入。
3. **budget 约束**：涉及补贴/预算分析时必须在工具参数里给出 budget（来自用户请求）。
4. **何时完成**：当结果足以回答用户问题时，直接输出最终报告（Markdown），不再调用
任何工具。报告须包含：执行摘要、关键发现（引用工具产出的具体 KPI 数字）、
策略建议（可执行、有优先级）、风险评估。
5. **诚实原则**：所有数字必须来自工具结果，禁止编造指标；工具失败时如实说明
失败范围与替代方案。

## 当前上下文
- 用户查询: {query}
- 预算: {budget}
- 数据路径: {data_path}
"""


def _build_system_prompt(state: dict[str, Any]) -> str:
    from src.agent.tools import build_tool_specs

    tool_descriptions = "\n".join(f"- `{s.name}`: {s.description}" for s in build_tool_specs())
    budget = state.get("budget")
    return CHIEF_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions,
        query=state.get("query", ""),
        budget=f"{budget:,.0f} CNY" if budget else "未指定",
        data_path=state.get("data_path") or "（无外部数据，使用合成演示数据）",
    )


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


async def chief_agent_node(state: AgentState) -> dict:
    """One step of the agent loop: LLM + bound tools decides the next action.

    Appends the AIMessage (with or without tool_calls) to ``messages``.
    """
    from src.agent.tools import build_tool_specs, to_langchain_tools
    from src.core.llm_factory import create_llm

    llm = create_llm(tier="power", agent_name="chief_agent")
    bound = llm.bind_tools(to_langchain_tools(build_tool_specs()))

    system = SystemMessage(content=_build_system_prompt(state))
    history = list(state.get("messages", []))
    # The user query leads the conversation (system prompt carries context).
    if not any(isinstance(m, HumanMessage) for m in history):
        history = [HumanMessage(content=state.get("query", "")), *history]

    response = await bound.ainvoke([system, *history])
    logger.info(
        "[chief_agent] tool_calls=%s",
        [tc.get("name") for tc in (response.tool_calls or [])] or "(final answer)",
    )
    return {"messages": [response], "status": AnalysisStatus.EXECUTING}


# ---------------------------------------------------------------------------
# Tools node
# ---------------------------------------------------------------------------


async def agent_tools_node(state: AgentState) -> dict:
    """Execute all tool_calls from the latest AIMessage.

    - Expert tools run in parallel worker threads (each via BaseAgent.run()
      with the full hook lifecycle and a per-expert timeout).
    - Full validated results merge into ``expert_results`` (reducer handles
      replacement); the LLM only receives compact digests as ToolMessages.
    """
    from src.agent.tools import execute_tool_call_async

    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage):
        return {"messages": []}

    tool_calls = last.tool_calls or []

    async def _one(tc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        args = tc.get("args") or {}
        digest = await execute_tool_call_async(state, tc.get("name", ""), dict(args))
        return tc, digest

    pairs = await asyncio.gather(*[_one(tc) for tc in tool_calls])

    tool_messages: list[ToolMessage] = []
    expert_results: list[dict[str, Any]] = []
    execution_errors: list[str] = []

    for tc, digest in pairs:
        name = tc.get("name", "")
        full = digest.pop("_full_result", None)
        if full is not None:
            # Merge hook-injected private metadata into the full result for
            # state, but strip it from what the LLM sees (already is).
            expert_results.append(full)
            if full.get("success") is False:
                execution_errors.append(
                    f"{full.get('expert', name)}: {full.get('error', 'failed')}"
                )
        content = json.dumps(digest, ensure_ascii=False, default=str)[:4000]
        tool_messages.append(ToolMessage(content=content, tool_call_id=tc.get("id", ""), name=name))

    update: dict[str, Any] = {"messages": tool_messages}
    if expert_results:
        update["expert_results"] = expert_results
    if execution_errors:
        update["execution_errors"] = execution_errors
    return update


def should_continue(state: AgentState) -> str:
    """Router: continue the loop or move to the quality gate."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "evaluate"


# ---------------------------------------------------------------------------
# Quality gate (Evaluator-Optimizer on the agent's final answer)
# ---------------------------------------------------------------------------


async def evaluate_gate_node(state: AgentState) -> dict:
    """Score the agent's final answer; below threshold re-enters the loop."""
    from src.core.config import get_settings
    from src.core.evaluator import evaluate_expert_output

    settings = get_settings()
    query = state.get("query", "")
    final_text = _last_ai_text(state)

    score = await evaluate_expert_output("chief_agent", query, final_text, tier="fast")

    current_round = state.get("refinement_round", 0) or 0
    needs_refinement = score.overall < settings.quality_threshold

    logger.info(
        "[evaluate_gate] overall=%.2f needs_refinement=%s round=%d",
        score.overall,
        needs_refinement,
        current_round + 1 if needs_refinement else current_round,
    )

    update: dict[str, Any] = {
        "quality_scores": {"chief_agent": score.model_dump()},
        "needs_refinement": needs_refinement,
        "refinement_round": current_round + (1 if needs_refinement else 0),
    }
    if needs_refinement and current_round < settings.max_refinement_rounds:
        update["messages"] = [
            HumanMessage(
                content=(
                    f"质量评审反馈（第 {current_round + 1} 轮，总分 {score.overall:.2f}，"
                    f"阈值 {settings.quality_threshold}）：{score.reasoning}\n"
                    "请基于反馈改进你的最终报告，补充缺失的分析维度或数据支撑。"
                )
            )
        ]
    else:
        # Accept the answer: route to approval / report via needs_refinement=False
        update["needs_refinement"] = False
    return update


def _route_after_gate(state: AgentState) -> str:
    """Router after the quality gate: refine (back to agent) or finish."""
    if state.get("needs_refinement") and (state.get("refinement_round", 0) or 0) <= (_max_rounds()):
        # Round was just incremented; loop while within bounds.
        return "agent"
    return "approval"


def _max_rounds() -> int:
    from src.core.config import get_settings

    return get_settings().max_refinement_rounds


# ---------------------------------------------------------------------------
# Approval + report
# ---------------------------------------------------------------------------


async def agent_approval_node(state: AgentState) -> dict:
    """Human approval checkpoint (mirrors workflow-mode approval_node)."""
    from src.core.config import get_settings

    settings = get_settings()
    budget = state.get("budget") or 0
    if not state.get("approval_required", False):
        return {"approved": True, "status": AnalysisStatus.REPORTING}
    if budget < settings.budget_approval_threshold:
        return {"approved": True, "status": AnalysisStatus.REPORTING}
    return {"approved": None, "status": AnalysisStatus.AWAITING_APPROVAL}


def _route_after_approval(state: AgentState) -> str:
    if state.get("approved") is True:
        return "report"
    return "__end__"


def _last_ai_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and not (getattr(m, "tool_calls", None)):
            return str(m.content)
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return str(m.content)
    return ""


async def agent_report_node(state: AgentState) -> dict:
    """Final report = agent's last answer (output-guarded) + memory write-back."""
    from src.core.token_tracker import get_cost_tracker
    from src.guardrails.output_guard import validate_output

    final_report = _last_ai_text(state)

    guard = validate_output(final_report)
    if not guard.passed:
        logger.warning("[agent_report] Output guardrail rejected: %s", guard.reason)
        final_report = (
            f"# Analysis Report\n\n## Query\n{state.get('query', '')}\n\n"
            f"*Report filtered by output guardrail: {guard.reason}.*\n"
        )

    cost_report = get_cost_tracker().report()

    # Memory write-back: close the loop (search_memory read it, this stores it)
    try:
        from src.core.config import get_settings
        from src.memory.manager import MemoryManager

        settings = get_settings()
        memory = MemoryManager(
            base_path=settings.memory_base_path,
            org_id=state.get("org_id", "default"),
        )
        memory.store(
            run_id=f"agent:{state.get('user_id', 'anonymous')}",
            query=state.get("query", ""),
            scope=state.get("scope") or "full",
            results_summary=final_report[:2000],
        )
    except Exception as exc:
        logger.debug("[agent_report] Memory write-back unavailable: %s", exc)

    return {
        "final_report": final_report,
        "token_usage": {
            "input_tokens": cost_report["input_tokens"],
            "output_tokens": cost_report["output_tokens"],
            "total_tokens": cost_report["total_tokens"],
        },
        "cost_usd": cost_report["total_cost_usd"],
        "status": AnalysisStatus.COMPLETED,
    }
