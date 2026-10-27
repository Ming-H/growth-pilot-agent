"""Chief Agent nodes for the agent-led orchestration mode.

Implements the classic agent loop on top of LangGraph:

    plan → agent ⇄ tools → evaluate ─┬→ agent (feedback, bounded rounds)
                                     └→ approval → report → kpi_gate

v9 planning pillar: ``agent_plan_node`` produces a structured subgoal list
(validated deterministically — the validator only deletes, never invents);
the subgoal checklist rides along in the chief's system prompt with live
status. With no LLM the node degrades to keyword routing (plan:rule) and
``chief_agent_node`` runs the **heuristic brain**: it synthesises the
tool_calls for each pending subgoal itself, so both brains traverse the
same graph. A hard step cap (``agent_max_steps``) bounds weak-model loops.

Every LLM decision point has a deterministic twin, and every degradation
is visible in ``state["degradations"]``.
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

_EXPERT_WHITELIST = ("prospect", "conversion", "subsidy", "retention", "ad")
_MAX_SUBGOALS = 5


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CHIEF_SYSTEM_PROMPT = """\
你是 GrowthPilot 的首席增长 Agent（Chief Agent），负责直接指挥一组领域专家工具完成用户增长分析。

## 可用工具
{tool_descriptions}

## 子目标清单（规划阶段产出；完成后逐项打勾）
{subgoal_checklist}

## 工作规则
1. **先思考再调用**：分析用户问题与子目标清单，判断需要哪些专家视角；若问题可能与本组织
历史分析相关，先调用 search_memory。
2. **按需调用**：你可以多次调用工具、观察结果后决定下一步——这是你的核心职责。
不同专家可以并行覆盖（分别调用），也可以串联深入。
3. **budget 约束**：涉及补贴/预算分析时必须在工具参数里给出 budget（来自用户请求）。
4. **何时完成**：当子目标全部完成（或明确失败）且结果足以回答用户问题时，直接输出最终
报告（Markdown），不再调用任何工具。报告须包含：执行摘要、关键发现（引用工具产出的
具体 KPI 数字）、策略建议（可执行、有优先级）、风险评估。
5. **诚实原则**：所有数字必须来自工具结果，禁止编造指标；工具失败时如实说明
失败范围与替代方案。

## 当前上下文
- 用户查询: {query}
- 预算: {budget}
- 数据路径: {data_path}
"""


def _render_subgoal_checklist(subgoals: list[dict[str, Any]]) -> str:
    if not subgoals:
        return "（无——由你自主规划）"
    icons = {"pending": "☐", "done": "☑", "failed": "☒"}
    lines = []
    for i, sg in enumerate(subgoals, 1):
        icon = icons.get(sg.get("status", "pending"), "☐")
        budget_note = f"（budget={sg['budget']:,.0f}）" if sg.get("budget") else ""
        lines.append(f"{i}. {icon} `{sg.get('expert')}`: {sg.get('question', '')}{budget_note}")
    return "\n".join(lines)


def _build_system_prompt(state: dict[str, Any]) -> str:
    from src.agent.tools import build_tool_specs

    tool_descriptions = "\n".join(f"- `{s.name}`: {s.description}" for s in build_tool_specs())
    budget = state.get("budget")
    return CHIEF_SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions,
        subgoal_checklist=_render_subgoal_checklist(state.get("subgoals") or []),
        query=state.get("query", ""),
        budget=f"{budget:,.0f} CNY" if budget else "未指定",
        data_path=state.get("data_path") or "（无外部数据，使用合成演示数据）",
    )


# ---------------------------------------------------------------------------
# Plan node (v9 planning pillar)
# ---------------------------------------------------------------------------

PLAN_SUBGOALS_PROMPT = """\
你是增长分析规划器。把用户问题拆解为至多 {max_subgoals} 个子目标，每个子目标指定一个专家。

可选专家: prospect / conversion / subsidy / retention / ad

用户问题: {query}
预算: {budget}

只输出 JSON（不要其他内容）:
{{"subgoals": [{{"expert": "<expert_key>", "question": "<该专家要回答的具体问题>", \
"budget": <可选，仅补贴相关>}}]}}
"""


def _validate_subgoals(raw: Any, query: str, budget: float | None) -> list[dict[str, Any]]:
    """Deterministic validator — 只删不造 (only deletes, never invents).

    Whitelist intersection, per-expert dedupe, cap, empty-question drop.
    The prior (keyword routing) is the FALLBACK, not an addition — a
    validator that could add experts would let code invent work the LLM
    never asked for.
    """
    if not isinstance(raw, dict):
        return []
    items = raw.get("subgoals")
    if not isinstance(items, list):
        return []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        expert = item.get("expert")
        question = str(item.get("question") or "").strip()
        if expert not in _EXPERT_WHITELIST or expert in seen or not question:
            continue
        seen.add(expert)
        sg: dict[str, Any] = {"expert": expert, "question": question[:300], "status": "pending"}
        b = item.get("budget", budget) if expert == "subsidy" else None
        if b:
            sg["budget"] = float(b)
        out.append(sg)
        if len(out) >= _MAX_SUBGOALS:
            break
    return out


async def agent_plan_node(state: AgentState) -> dict:
    """Structured subgoal planning before the agent loop.

    LLM path: role=fast JSON plan → deterministic validator (只删不造).
    Zero-key path: keyword routing prior (plan:rule) — the same graph,
    visibly degraded.
    """
    import time as _time

    started = _time.monotonic()
    query = state.get("query", "")
    budget = state.get("budget")

    from src.core.llm_factory import create_llm

    llm = create_llm(role="fast", agent_name="plan_node")
    subgoals: list[dict[str, Any]] = []
    source = "rule"

    if llm is not None:
        try:
            prompt = PLAN_SUBGOALS_PROMPT.format(
                max_subgoals=_MAX_SUBGOALS, query=query, budget=budget or "未指定"
            )
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            subgoals = _validate_subgoals(json.loads(content), query, budget)
            if subgoals:
                source = "llm"
        except Exception as exc:
            logger.warning("[plan] LLM planning failed, keyword prior: %s", exc)

    if not subgoals:
        # Keyword prior (the same router the workflow mode trusts).
        from src.graph.nodes import _intent_classify

        selected = _intent_classify(query)
        subgoals = [
            {
                "expert": key,
                "question": query,
                **({"budget": float(budget)} if key == "subsidy" and budget else {}),
                "status": "pending",
            }
            for key in selected[:_MAX_SUBGOALS]
        ]
        source = "rule"

    selected_experts = [sg["expert"] for sg in subgoals]
    degradations: list[str] = [] if source == "llm" else ["plan:rule"]
    dur_ms = round((_time.monotonic() - started) * 1000)
    logger.info("[plan] source=%s subgoals=%s (%dms)", source, selected_experts, dur_ms)
    return {
        "subgoals": subgoals,
        "selected_experts": selected_experts,
        "plan": {"source": source, "subgoals": subgoals},
        "degradations": degradations,
        "trace": [
            {"node": "plan", "dur_ms": dur_ms, "detail": f"source={source} n={len(subgoals)}"}
        ],
        "status": AnalysisStatus.EXECUTING,
    }


# ---------------------------------------------------------------------------
# Agent node (LLM brain ⇄ heuristic brain on the same graph)
# ---------------------------------------------------------------------------


def _pending_subgoals(state: AgentState) -> list[dict[str, Any]]:
    return [sg for sg in (state.get("subgoals") or []) if sg.get("status") == "pending"]


def _heuristic_brain_message(state: AgentState) -> AIMessage:
    """Synthesise the tool_calls for every pending subgoal (zero-key brain).

    The graph then flows through the unchanged tools node — both brains
    traverse the same topology; only the decision-maker differs.
    """
    tool_calls = []
    for i, sg in enumerate(_pending_subgoals(state)):
        args: dict[str, Any] = {"query": sg.get("question", state.get("query", ""))}
        if sg.get("budget"):
            args["budget"] = sg["budget"]
        tool_calls.append(
            {"name": f"run_{sg['expert']}_analysis", "args": args, "id": f"heuristic-{i}"}
        )
    if tool_calls:
        return AIMessage(content="", tool_calls=tool_calls)
    # Nothing pending → deterministic final-answer placeholder (the report
    # node replaces it with the template when it is too short).
    return AIMessage(content="# 启发式大脑汇总\n\n（确定性模式：见 KPI 快照与结构化结果）")


async def chief_agent_node(state: AgentState) -> dict:
    """One step of the agent loop: LLM + bound tools decides the next action.

    - LLM present: normal tool-calling decision.
    - LLM absent: heuristic brain — issue all pending subgoals as one batch.
    - Step budget exceeded: wrap up (no more tool calls) so the loop ends.
    """
    import time as _time

    from src.core.config import get_settings
    from src.core.llm_factory import create_llm

    started = _time.monotonic()
    settings = get_settings()
    step_count = len(state.get("agent_steps") or [])
    degradations: list[str] = []

    # Step budget: force a final answer on the next LLM turn.
    budget_note = ""
    if step_count >= settings.agent_max_steps:
        degradations.append("agent_loop:capped")
        budget_note = "\n\n[系统] 步数预算已耗尽，请立即输出最终报告，不要再调用工具。"
        logger.warning("[chief_agent] step budget reached (%d)", step_count)

    llm = create_llm(role="power", agent_name="chief_agent")

    if llm is None:
        # Heuristic brain: pending subgoals as one batch; when nothing is
        # pending, produce the (deterministic) final answer marker.
        pending = _pending_subgoals(state)
        if pending and step_count < settings.agent_max_steps:
            response = _heuristic_brain_message(state)
            thought = f"heuristic brain: dispatch {len(pending)} pending subgoals"
        else:
            response = AIMessage(
                content="# 启发式大脑汇总\n\n（确定性模式：见 KPI 快照与结构化结果）"
            )
            thought = "heuristic brain: no pending subgoals — final answer"
        dur_ms = round((_time.monotonic() - started) * 1000)
        update: dict[str, Any] = {
            "messages": [response],
            "status": AnalysisStatus.EXECUTING,
            "agent_steps": [
                {
                    "thought": thought,
                    "action": [tc.get("name") for tc in (response.tool_calls or [])] or "final",
                    "obs": "-",
                    "gate": "heuristic",
                }
            ],
            "trace": [{"node": "agent", "dur_ms": dur_ms, "detail": thought}],
        }
        if degradations:
            update["degradations"] = degradations
        return update

    from src.agent.tools import build_tool_specs, to_langchain_tools

    bound = llm.bind_tools(to_langchain_tools(build_tool_specs()))

    system = SystemMessage(content=_build_system_prompt(state) + budget_note)
    history = list(state.get("messages", []))
    # The user query leads the conversation (system prompt carries context).
    if not any(isinstance(m, HumanMessage) for m in history):
        history = [HumanMessage(content=state.get("query", "")), *history]

    response = await bound.ainvoke([system, *history])

    # Weak-model flip (chief_tools:weak): a local small model whose first
    # turns produce no valid tool calls is escorted to the heuristic brain —
    # pending subgoals get dispatched deterministically instead of looping.
    if settings.channel == "local" and not response.tool_calls:
        prior_llm_steps = [s for s in (state.get("agent_steps") or []) if s.get("gate") == "llm"]
        if len(prior_llm_steps) >= 1 and _pending_subgoals(state):
            logger.warning(
                "[chief_agent] local model produced no tool calls (%d steps) — "
                "switching to heuristic brain",
                len(prior_llm_steps) + 1,
            )
            degradations.append("chief_tools:weak")
            response = _heuristic_brain_message(state)
            if not response.tool_calls:
                response = AIMessage(
                    content="# 启发式大脑汇总\n\n（确定性模式：见 KPI 快照与结构化结果）"
                )
    thought = str(response.content or "")[:160]
    dur_ms = round((_time.monotonic() - started) * 1000)
    logger.info(
        "[chief_agent] tool_calls=%s",
        [tc.get("name") for tc in (response.tool_calls or [])] or "(final answer)",
    )
    update = {
        "messages": [response],
        "status": AnalysisStatus.EXECUTING,
        "agent_steps": [
            {
                "thought": thought,
                "action": [tc.get("name") for tc in (response.tool_calls or [])] or "final",
                "obs": "-",
                "gate": "llm",
            }
        ],
        "trace": [
            {
                "node": "agent",
                "dur_ms": dur_ms,
                "detail": "tool_calls="
                + str([tc.get("name") for tc in (response.tool_calls or [])]),
            }
        ],
    }
    if degradations:
        update["degradations"] = degradations
    return update


# ---------------------------------------------------------------------------
# Tools node
# ---------------------------------------------------------------------------


def _shrink_digest(digest: dict[str, Any]) -> dict[str, Any]:
    """Field-wise truncation — never slice the JSON string itself."""
    out = dict(digest)
    if isinstance(out.get("summary"), str) and len(out["summary"]) > 400:
        out["summary"] = out["summary"][:400] + "…"
    if isinstance(out.get("kpis"), dict) and len(out["kpis"]) > 20:
        out["kpis"] = dict(list(out["kpis"].items())[:20])
    if isinstance(out.get("errors"), list) and len(out["errors"]) > 3:
        out["errors"] = out["errors"][:3] + ["…"]
    return out


async def agent_tools_node(state: AgentState) -> dict:
    """Execute all tool_calls from the latest AIMessage.

    - Expert tools run in parallel worker threads (each via BaseAgent.run()
      with the full hook lifecycle and a per-expert timeout).
    - Full validated results merge into ``expert_results`` (reducer handles
      replacement); the LLM only receives compact digests as ToolMessages.
    - Subgoal statuses update from the results (checkpoint discipline:
      downstream never consumes unverified upstream output).
    """
    import time as _time

    started = _time.monotonic()
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
    obs_notes: list[str] = []
    subgoal_updates: list[dict[str, Any]] = []

    for tc, digest in pairs:
        name = tc.get("name", "")
        full = digest.pop("_full_result", None)
        if full is not None:
            # Merge hook-injected private metadata into the full result for
            # state, but strip it from what the LLM sees (already is).
            expert_results.append(full)
            ok = full.get("success") is not False and bool(full.get("kpis"))
            obs_notes.append(f"{full.get('expert', name)}:{'ok' if ok else 'failed'}")
            if full.get("success") is False:
                execution_errors.append(
                    f"{full.get('expert', name)}: {full.get('error', 'failed')}"
                )
            # Subgoal checkpoint: kpis present ∧ success → done, else failed.
            expert_key = str(full.get("expert", "") or "")
            if expert_key:
                subgoal_updates.append(
                    {
                        "expert": expert_key,
                        "question": tc.get("args", {}).get("query", ""),
                        "budget": tc.get("args", {}).get("budget"),
                        "status": "done" if ok else "failed",
                    }
                )
        content = json.dumps(_shrink_digest(digest), ensure_ascii=False, default=str)
        tool_messages.append(ToolMessage(content=content, tool_call_id=tc.get("id", ""), name=name))

    update: dict[str, Any] = {"messages": tool_messages}
    if expert_results:
        update["expert_results"] = expert_results
    if execution_errors:
        update["execution_errors"] = execution_errors
    if subgoal_updates:
        update["subgoals"] = subgoal_updates
    # KPI snapshot refresh (ground truth for the gate).
    snapshot: dict[str, Any] = {}
    for r in expert_results:
        k = r.get("kpis")
        if isinstance(k, dict):
            snapshot.update(k)
    if snapshot:
        merged = dict(state.get("kpi_snapshot") or {})
        merged.update(snapshot)
        update["kpi_snapshot"] = merged
    dur_ms = round((_time.monotonic() - started) * 1000)
    update["agent_steps"] = [
        {
            "thought": "-",
            "action": [tc.get("name") for tc in tool_calls],
            "obs": "; ".join(obs_notes)[:200],
            "gate": "tools",
        }
    ]
    update["trace"] = [{"node": "tools", "dur_ms": dur_ms, "detail": "; ".join(obs_notes)[:120]}]
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


def _deterministic_score(state: AgentState) -> tuple[float, str]:
    """Zero-key scoring: subgoal completion ratio / KPI coverage."""
    subgoals = state.get("subgoals") or []
    done = sum(1 for sg in subgoals if sg.get("status") == "done")
    failed = sum(1 for sg in subgoals if sg.get("status") == "failed")
    completion = (done / len(subgoals)) if subgoals else 0.0
    kpis = len(state.get("kpi_snapshot") or {})
    coverage = min(1.0, kpis / max(len(subgoals), 1)) if subgoals else 0.0
    overall = round(0.6 * completion + 0.4 * coverage, 2)
    reasoning = (
        f"deterministic scoring (no LLM): {done}/{len(subgoals)} subgoals done"
        f"{'，' + str(failed) + ' failed' if failed else ''}, {kpis} KPIs"
    )
    return overall, reasoning


async def evaluate_gate_node(state: AgentState) -> dict:
    """Score the agent's final answer; below threshold re-enters the loop."""
    import time as _time

    from src.core.config import get_settings
    from src.core.evaluator import evaluate_expert_output

    started = _time.monotonic()
    settings = get_settings()
    query = state.get("query", "")
    final_text = _last_ai_text(state)

    from src.core.llm_factory import create_resilient_llm

    judge = create_resilient_llm(role="fast", agent_name="evaluator")

    degradations: list[str] = []
    if judge is None:
        overall, reasoning = _deterministic_score(state)
        score_dict = {
            "completeness": overall,
            "actionability": overall,
            "data_grounding": 1.0 if state.get("kpi_snapshot") else 0.3,
            "overall": overall,
            "reasoning": reasoning,
            "evaluation_mode": "rule",
        }
        needs_refinement = False  # no brain ⇒ refine cannot improve anything
        degradations = ["eval:rule"]
        logger.info("[evaluate_gate] eval:rule overall=%.2f (%s)", overall, reasoning)
    else:
        score = await evaluate_expert_output("chief_agent", query, final_text, tier="fast")
        score_dict = score.model_dump()
        needs_refinement = score.overall < settings.quality_threshold
        logger.info(
            "[evaluate_gate] overall=%.2f needs_refinement=%s",
            score.overall,
            needs_refinement,
        )

    current_round = state.get("refinement_round", 0) or 0

    update: dict[str, Any] = {
        "quality_scores": {"chief_agent": score_dict},
        "needs_refinement": needs_refinement,
        "refinement_round": current_round + (1 if needs_refinement else 0),
        "trace": [
            {
                "node": "evaluate",
                "dur_ms": round((_time.monotonic() - started) * 1000),
                "detail": f"overall={score_dict.get('overall')}",
            }
        ],
    }
    if degradations:
        update["degradations"] = degradations
    if needs_refinement and current_round < settings.max_refinement_rounds:
        update["messages"] = [
            HumanMessage(
                content=(
                    f"质量评审反馈（第 {current_round + 1} 轮，总分 {score_dict['overall']:.2f}，"
                    f"阈值 {settings.quality_threshold}）：{score_dict.get('reasoning', '')}\n"
                    "请基于反馈改进你的最终报告，补充缺失的分析维度或数据支撑。"
                )
            )
        ]
    else:
        # Accept the answer: route to approval / report via needs_refinement=False
        update["needs_refinement"] = False
    return update


def _route_after_gate(state: AgentState) -> str:
    """Router after the quality gate: refine (back to agent) or finish.

    A7 fix: `<` not `<=` — the round was already incremented by the gate,
    so comparing with `<=` allowed one extra regeneration beyond the bound
    (workflow mode always used `<`).
    """
    if state.get("needs_refinement") and (state.get("refinement_round", 0) or 0) < _max_rounds():
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
    """Final report = agent's last answer (output-guarded).

    With no expert results at all (heuristic brain ran dry), degrade to the
    deterministic template so a report is always produced (report:template).
    The kpi_gate node handles reconciliation + memory write-back.
    """
    import time as _time

    from src.core.token_tracker import get_cost_tracker
    from src.guardrails.output_guard import validate_output

    started = _time.monotonic()
    final_report = _last_ai_text(state)
    degradations: list[str] = []

    # Heuristic brain's placeholder (or an empty answer) → template report.
    if not state.get("expert_results") or len(final_report) < 50:
        try:
            from src.report.generator import ReportGenerator

            final_report = ReportGenerator().generate_report(dict(state))
            degradations.append("report:template")
            logger.info("[agent_report] report:template (no usable agent answer)")
        except Exception as exc:
            logger.warning("[agent_report] template render failed: %s", exc)

    guard = validate_output(final_report)
    if not guard.passed:
        logger.warning("[agent_report] Output guardrail rejected: %s", guard.reason)
        final_report = (
            f"# Analysis Report\n\n## Query\n{state.get('query', '')}\n\n"
            f"*Report filtered by output guardrail: {guard.reason}.*\n"
        )

    cost_report = get_cost_tracker().report()

    # KPI snapshot from expert results (single aggregation point)
    kpi_snapshot: dict[str, Any] = {}
    for r in state.get("expert_results", []):
        ek = r.get("kpis")
        if isinstance(ek, dict):
            kpi_snapshot.update(ek)

    update = {
        "final_report": final_report,
        "kpi_snapshot": kpi_snapshot,
        "token_usage": {
            "input_tokens": cost_report["input_tokens"],
            "output_tokens": cost_report["output_tokens"],
            "total_tokens": cost_report["total_tokens"],
        },
        "cost_usd": cost_report["total_cost_usd"],
        "status": AnalysisStatus.COMPLETED,
        "trace": [
            {
                "node": "report",
                "dur_ms": round((_time.monotonic() - started) * 1000),
                "detail": "chars=%d" % len(final_report),
            }
        ],
    }
    if degradations:
        update["degradations"] = degradations
    return update
