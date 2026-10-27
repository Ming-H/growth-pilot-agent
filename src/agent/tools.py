"""Tools exposed to the Chief Agent in the agent-led orchestration mode.

Design (Agent-as-Tool):
- Each domain expert becomes a callable tool the LLM can invoke dynamically.
- Tool handlers always route through ``BaseAgent.run()`` so the hook
  lifecycle (tracing/logging/metrics) and middleware stack execute, exactly
  as in workflow mode.
- Full expert results are written into the graph state (via the node, not
  the LLM) — the tool *returns* a compact digest (kpis + summary) to keep
  the agent's context window small.
- ``search_memory`` gives the agent read access to per-org cross-session
  memory; writes happen automatically after a successful run.

Tool specs are (schema, handler) pairs: ``to_langchain_tools`` renders the
schemas for ``bind_tools``; ``execute_tool_call`` dispatches to the handler
with access to graph state (which plain LangChain tools cannot carry).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument schemas (what the LLM sees and fills)
# ---------------------------------------------------------------------------


class ExpertToolArgs(BaseModel):
    """Arguments for a domain-expert tool call."""

    query: str = Field(
        description="The specific analytical question for this expert (Chinese or English)"
    )
    budget: float | None = Field(
        default=None,
        description="Budget in CNY if relevant to the question (required for subsidy analysis)",
    )


class MemorySearchArgs(BaseModel):
    """Arguments for the memory search tool."""

    query: str = Field(
        description="Search text for retrieving relevant prior analyses of this organisation"
    )


# ---------------------------------------------------------------------------
# Tool spec model
# ---------------------------------------------------------------------------


class ToolSpec:
    """A tool exposed to the Chief Agent: schema + state-aware handler.

    ACI discipline: a tool spec is the contract between the LLM and the
    action space — it must say what the tool is for, when to call it, and
    how it fails, so the model can self-correct without a stack trace.
    """

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        handler: Callable[..., Any],
        failure_modes: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.handler = handler
        self.failure_modes = failure_modes or []


# ---------------------------------------------------------------------------
# Expert tool implementation (shared)
# ---------------------------------------------------------------------------

# Keys of the compact digest returned to the LLM (everything else stays in
# graph state only, keeping the agent context small).
_DIGEST_KEYS = ("expert", "success", "kpis", "errors")


def _run_expert_tool(
    state: dict[str, Any],
    expert_key: str,
    query: str,
    budget: float | None,
) -> dict[str, Any]:
    """Execute one expert via its full BaseAgent lifecycle.

    Runs in a worker thread (see ``execute_tool_call_async``) so a fresh
    event loop is available for ``asyncio.run``.
    Returns the compact digest for the LLM; the *full* validated result is
    carried under ``_full_result`` for the tools node to merge into state.
    """
    from src.core.config import get_settings
    from src.graph.nodes import _create_expert, _validate_expert_result

    params: dict[str, Any] = {
        "query": query,
        "scope": expert_key,
        "budget": budget or state.get("budget") or 0,
        "data_path": state.get("data_path") or "",
        "memory_context": [],  # the agent decides memory usage via search_memory
    }
    settings = get_settings()
    expert = _create_expert(expert_key)

    async def _invoke() -> dict[str, Any]:
        async with asyncio.timeout(settings.expert_timeout_seconds):
            return await expert.run({"params": params})

    result = asyncio.run(_invoke())
    result["expert"] = expert_key
    result = _validate_expert_result(expert_key, result)

    digest = {k: v for k, v in result.items() if k in _DIGEST_KEYS}
    summary = (
        (result.get("analysis") or {}).get("summary")
        if isinstance(result.get("analysis"), dict)
        else None
    )
    if summary:
        digest["summary"] = str(summary)[:400]
    digest["_full_result"] = result  # stripped by the tools node before state merge
    return digest


async def _search_memory_tool(state: dict[str, Any], query: str) -> dict[str, Any]:
    """Retrieve prior analyses + verified facts for this org (best-effort).

    Episodic search over past analyses plus the L1 fact ledger — numbers
    the agent may cite (the KPI gate still verifies the final report).
    """
    try:
        from src.core.config import get_settings
        from src.memory import MemoryStore

        store = MemoryStore(db_path=get_settings().memory_db_path)
        org_id = state.get("org_id", "default")
        hits = await store.search_analyses(org_id, query, limit=3)
        facts = await store.search_facts(org_id, limit=10)
        await store.close()
        return {
            "found": len(hits),
            "entries": [
                {
                    "scope": h.get("scope", ""),
                    "query": str(h.get("query", ""))[:120],
                    "summary": str(h.get("results_summary", ""))[:300],
                }
                for h in hits
            ],
            "verified_facts": [
                {"domain": f["domain"], "metric": f["metric_key"], "value": f["value"]}
                for f in facts[:10]
            ],
        }
    except Exception as exc:  # memory must never break the agent loop
        logger.debug("[search_memory] unavailable: %s", exc)
        return {"found": 0, "entries": [], "note": "memory unavailable"}


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------

# ACI-style self-describing tool metadata: purpose / when-to-call /
# required-args / failure-modes. The failure-modes section is what lets the
# chief agent interpret a failed call without seeing a stack trace.
EXPERT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "prospect": (
        "【用途】潜客识别：用户意向预测(LightGBM)、评分排序、RFM 分群、LTV 预测(BG/NBD)。"
        "【何时调用】问题涉及拉新、高潜用户、用户价值、分群时。"
        "【参数】query=具体分析问题。"
        "【失败模式】特征数据缺失时退化为合成演示数据（结果标注 data_source）；"
        "LTV 拟合不收敛时返回启发式估值。"
    ),
    "conversion": (
        "【用途】转化优化：漏斗分析、触达规划、投放位分配、优惠券设计、多触点归因。"
        "【何时调用】问题涉及转化率、漏斗瓶颈、触达策略、归因时。"
        "【参数】query=具体分析问题。"
        "【失败模式】漏斗数据为空时返回空结果并说明；预算不足时优惠券设计收缩规模。"
    ),
    "subsidy": (
        "【用途】补贴策略：因果推断 ATE/CATE(statsmodels 后门调整/IPW)、价格弹性、"
        "整数规划预算优化(HiGHS)。"
        "【何时调用】问题涉及补贴效果、预算分配、ROI 时。"
        "【参数】query + budget（必须提供，单位 CNY）。"
        "【失败模式】无实验数据时 ATE 退化为历史经验值并标注 heuristic；"
        "求解器不可用时退化为贪心分配（method 字段标明）。"
    ),
    "retention": (
        "【用途】留存召回：流失预警(LightGBM)、同期群分析、新客养成、流失召回策略。"
        "【何时调用】问题涉及流失、留存、召回、用户生命周期时。"
        "【参数】query=具体分析问题。"
        "【失败模式】数据不足时流失模型 AUC 不可靠（结果带 train_auc 供校验）；"
        "召回计划退化为固定分层模板。"
    ),
    "ad": (
        "【用途】广告投放：RTA 策略、OCPX 出价优化、创意疲劳检测、受众定向。"
        "【何时调用】问题涉及投放、出价、CPA、创意、受众扩展时。"
        "【参数】query=具体分析问题。"
        "【失败模式】当前版本始终使用内置合成数据（data_source=synthetic，"
        "忽略 data_path）——接入真实广告平台数据在路线图中。"
    ),
}

EXPERT_TOOL_FAILURE_MODES: dict[str, list[str]] = {
    "prospect": ["synthetic_data_fallback", "ltv_fit_degenerate"],
    "conversion": ["empty_funnel_data", "budget_exhausted"],
    "subsidy": ["heuristic_ate_fallback", "greedy_allocation_fallback"],
    "retention": ["low_auc_warning", "template_winback_fallback"],
    "ad": ["synthetic_data_only"],
}


def build_tool_specs() -> list[ToolSpec]:
    """Build the tool specs exposed to the Chief Agent."""
    specs: list[ToolSpec] = []
    for expert_key, description in EXPERT_TOOL_DESCRIPTIONS.items():
        specs.append(
            ToolSpec(
                name=f"run_{expert_key}_analysis",
                description=description,
                args_schema=ExpertToolArgs,
                handler=lambda state, query, budget=None, _k=expert_key: _run_expert_tool(
                    state, _k, query, budget
                ),
                failure_modes=EXPERT_TOOL_FAILURE_MODES.get(expert_key),
            )
        )
    specs.append(
        ToolSpec(
            name="search_memory",
            description=(
                "检索本组织的历史分析记忆（跨会话上下文）。"
                "当当前问题可能与既往分析相关时先调用此工具。"
            ),
            args_schema=MemorySearchArgs,
            handler=_search_memory_tool,
        )
    )
    return specs


def to_langchain_tools(specs: list[ToolSpec]) -> list[BaseTool]:
    """Render specs as LangChain tools (schema only) for ``bind_tools``."""

    def _make_factory(spec: ToolSpec) -> Callable[..., str]:
        def _noop(**kwargs: Any) -> str:
            return f"{spec.name}: executed by the graph tools node"

        return _noop

    from langchain_core.tools import StructuredTool

    return [
        StructuredTool.from_function(
            func=_make_factory(spec),
            name=spec.name,
            description=spec.description,
            args_schema=spec.args_schema,
        )
        for spec in specs
    ]


def get_spec_by_name(name: str) -> ToolSpec | None:
    for spec in build_tool_specs():
        if spec.name == name:
            return spec
    return None


async def execute_tool_call_async(
    state: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a tool call with graph-state access (async entry point)."""
    spec = get_spec_by_name(tool_name)
    if spec is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        # Expert tools are IO/LLM bound — run handler in a worker thread.
        result = await asyncio.to_thread(
            _run_handler_with_event_loop, spec.handler, state, tool_args
        )
        return result
    except Exception as exc:
        logger.warning("[tools] %s failed: %s", tool_name, exc)
        return {"error": f"{tool_name} failed: {exc}"}


def _run_handler_with_event_loop(
    handler: Callable[..., Any],
    state: dict[str, Any],
    tool_args: dict[str, Any],
) -> Any:
    """Run a possibly-async handler in a fresh event loop (worker thread)."""
    outcome = handler(state, **tool_args)
    if asyncio.iscoroutine(outcome):
        return asyncio.run(outcome)
    return outcome
