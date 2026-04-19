"""OrchestratorAgent - Decides which sub-agents to run, dispatches, aggregates."""

from __future__ import annotations

import logging
from typing import Any

from src.core.base import BaseAgent
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-agent imports (lazy to avoid circular imports at module level)
# ---------------------------------------------------------------------------
from src.agents.prospect import ProspectAgent
from src.agents.conversion import ConversionAgent
from src.agents.subsidy import SubsidyAgent
from src.agents.retention import RetentionAgent
from src.agents.ad import AdAgent

SYSTEM_PROMPT = """\
你是 GrowthPilot 编排 Agent。你负责：
1. 理解用户查询意图
2. 决定需要执行哪些子 Agent
3. 编排子 Agent 的执行流程
4. 汇总各子 Agent 的分析结果
5. 生成 KPI 快照和策略建议

请根据用户查询和当前状态，输出 JSON 格式的编排决策和汇总分析。
"""

# Scope keywords for intent detection
_SCOPE_KEYWORDS: dict[str, list[str]] = {
    "prospect": ["潜客", "获客", "拉新", "新用户", "prospect", "acquisition", "评分", "画像"],
    "conversion": ["转化", "漏斗", "conversion", "funnel", "优惠券", "coupon", "归因", "attribution"],
    "subsidy": ["补贴", "优惠", "budget", "subsidy", "预算", "弹性", "elasticity", "ROI"],
    "retention": ["留存", "流失", "挽回", "retention", "churn", "winback", "nurture", "培育", "群组"],
    "ad": ["广告", "投放", "RTA", "出价", "bid", "创意", "creative", "受众", "audience", "ad"],
    "inapp": ["站内", "in-app", "inapp", "push", "推送", "消息"],
}


class OrchestratorAgent(BaseAgent):
    """Orchestrates sub-agents based on user query scope."""

    name = "orchestrator"
    description = "编排 Agent - 决定执行哪些子 Agent 并汇总结果"

    def __init__(self, llm: Any, system_prompt: str = SYSTEM_PROMPT) -> None:
        super().__init__(llm=llm, system_prompt=system_prompt)
        self._agents: dict[str, BaseAgent] = {
            "prospect": ProspectAgent(llm),
            "subsidy": SubsidyAgent(llm),
            "ad": AdAgent(llm),
            "conversion": ConversionAgent(llm),
            "retention": RetentionAgent(llm),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(self, state: AgentState) -> dict[str, Any]:
        """Orchestrate the analysis workflow.

        1. Detect scope from the user query.
        2. Dispatch relevant sub-agents.
        3. Aggregate results and build KPI snapshot.
        """
        errors: list[str] = []

        # 1. Detect scope
        scope = state.get("scope") or self._detect_scope(state.get("query", ""))
        agents_to_run = self._resolve_agents(scope)

        logger.info("Orchestrator scope=%s agents=%s", scope, agents_to_run)

        # 2. Run agents (parallel group first, then sequential)
        update: dict[str, Any] = {}

        # Phase 1: parallel agents (prospect, subsidy, ad)
        parallel_agents = [a for a in agents_to_run if a in ("prospect", "subsidy", "ad")]
        for agent_name in parallel_agents:
            try:
                agent_result = await self._run_agent(agent_name, state)
                update.update(agent_result)
                # Merge errors
                if agent_result.get("errors"):
                    errors.extend(agent_result["errors"])
            except Exception as exc:
                msg = f"Agent {agent_name} failed: {exc}"
                logger.error(msg)
                errors.append(msg)

        # Rebuild state with parallel results for sequential agents
        merged_state: AgentState = {**state, **update}  # type: ignore[typeddict-item]

        # Phase 2: conversion (depends on prospect + subsidy)
        if "conversion" in agents_to_run:
            try:
                agent_result = await self._run_agent("conversion", merged_state)
                update.update(agent_result)
                if agent_result.get("errors"):
                    errors.extend(agent_result["errors"])
            except Exception as exc:
                msg = f"Agent conversion failed: {exc}"
                logger.error(msg)
                errors.append(msg)

            merged_state = {**merged_state, **update}  # type: ignore[typeddict-item]

        # Phase 3: retention (depends on conversion)
        if "retention" in agents_to_run:
            try:
                agent_result = await self._run_agent("retention", merged_state)
                update.update(agent_result)
                if agent_result.get("errors"):
                    errors.extend(agent_result["errors"])
            except Exception as exc:
                msg = f"Agent retention failed: {exc}"
                logger.error(msg)
                errors.append(msg)

        # 3. Build KPI snapshot
        kpi_snapshot = self._build_kpi_snapshot({**state, **update})  # type: ignore[typeddict-item]
        update["kpi_snapshot"] = kpi_snapshot

        # 4. LLM summary
        try:
            prompt = self._build_orchestrator_prompt(update, state)
            llm_response = await self._invoke_llm(prompt)
            parsed = self._parse_json_response(llm_response)
            update["analysis_summary"] = parsed.get("summary", "")
            update["strategy_recommendation"] = parsed.get("strategy_recommendation", "")
        except Exception as exc:
            logger.warning("Orchestrator LLM synthesis failed: %s", exc)
            update["analysis_summary"] = "分析完成，但 LLM 摘要生成失败"
            update["strategy_recommendation"] = ""
            errors.append(f"Orchestrator LLM: {exc}")

        if errors:
            update["errors"] = errors

        update["metadata"] = [{"scope": scope, "agents_run": agents_to_run}]

        return update

    # ------------------------------------------------------------------
    # Scope detection
    # ------------------------------------------------------------------
    def _detect_scope(self, query: str) -> str:
        """Detect the analysis scope from a user query.

        Returns one of: ``full``, ``prospect``, ``conversion``, ``subsidy``,
        ``retention``, ``ad``, ``inapp``.
        """
        if not query:
            return "full"

        query_lower = query.lower()
        scores: dict[str, int] = {}

        for scope_name, keywords in _SCOPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[scope_name] = score

        if not scores:
            return "full"

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best

    # ------------------------------------------------------------------
    # Agent helpers
    # ------------------------------------------------------------------
    def _resolve_agents(self, scope: str) -> list[str]:
        """Map scope to list of agent names to execute."""
        mapping: dict[str, list[str]] = {
            "full": ["prospect", "subsidy", "ad", "conversion", "retention"],
            "inapp": ["conversion", "retention"],
            "prospect": ["prospect", "conversion"],
            "conversion": ["conversion"],
            "subsidy": ["subsidy"],
            "retention": ["retention"],
            "ad": ["ad"],
        }
        return mapping.get(scope, mapping["full"])

    async def _run_agent(self, name: str, state: AgentState) -> dict[str, Any]:
        """Run a single sub-agent and return its partial state update."""
        agent = self._agents.get(name)
        if agent is None:
            raise ValueError(f"Unknown agent: {name}")
        return await agent.run(state)

    # ------------------------------------------------------------------
    # KPI snapshot
    # ------------------------------------------------------------------
    def _build_kpi_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        """Aggregate key metrics from all sub-agent results."""
        prospect = state.get("prospect_results") or {}
        conversion = state.get("conversion_results") or {}
        subsidy = state.get("subsidy_results") or {}
        retention = state.get("retention_results") or {}
        ad = state.get("ad_results") or {}

        # Extract conversion rate from funnel_result
        funnel = conversion.get("funnel_result", {})
        overall_cvr = funnel.get("overall_conversion_rate", 0.0)

        # Extract churn risk ratio
        churn_risk = retention.get("churn_risk", {})
        high_risk_ratio = churn_risk.get("high_risk_ratio", 0.0) if isinstance(churn_risk, dict) else 0.0

        # Extract intent AUC
        intent_metrics = prospect.get("intent_metrics", {})
        intent_auc = intent_metrics.get("auc", 0.0) if isinstance(intent_metrics, dict) else 0.0

        return {
            "total_users": prospect.get("user_count", 0),
            "intent_auc": intent_auc,
            "conversion_rate": overall_cvr,
            "expected_roi": subsidy.get("expected_roi", 0.0),
            "churn_risk_ratio": high_risk_ratio,
            "ad_cpa": ad.get("expected_cpa", 0.0),
            "budget": state.get("budget", 0),
            "scope": state.get("scope", "full"),
        }

    @staticmethod
    def _extract_ratio(data: dict) -> float:
        """Try to extract a ratio/percentage from varied tool output shapes."""
        if not data:
            return 0.0
        for key in ("ratio", "rate", "percentage", "high_intent_ratio", "overall", "avg"):
            if key in data:
                val = data[key]
                if isinstance(val, (int, float)):
                    return float(val)
        return 0.0

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _build_orchestrator_prompt(self, update: dict[str, Any], state: AgentState) -> str:
        context = self._build_prompt_context(state)
        parts = [context, "", "## 各 Agent 分析结果摘要"]

        if update.get("prospect_results"):
            analysis = update["prospect_results"].get("analysis", {})
            parts.append(f"- 潜客识别: {analysis.get('summary', '已完成')}")
        if update.get("conversion_results"):
            analysis = update["conversion_results"].get("analysis", {})
            parts.append(f"- 转化策略: {analysis.get('summary', '已完成')}")
        if update.get("subsidy_results"):
            analysis = update["subsidy_results"].get("analysis", {})
            parts.append(f"- 补贴策略: {analysis.get('summary', '已完成')}")
        if update.get("retention_results"):
            analysis = update["retention_results"].get("analysis", {})
            parts.append(f"- 用户留存: {analysis.get('summary', '已完成')}")
        if update.get("ad_results"):
            analysis = update["ad_results"].get("analysis", {})
            parts.append(f"- 广告投放: {analysis.get('summary', '已完成')}")

        parts.append("""
请基于以上各 Agent 的分析结果，给出：
1. 整体分析摘要
2. 关键策略建议

请以 JSON 格式输出:
{
  "summary": "整体分析摘要 (3-5 句话)",
  "strategy_recommendation": "关键策略建议 (3-5 条具体建议，用分号分隔)"
}""")
        return "\n".join(parts)
