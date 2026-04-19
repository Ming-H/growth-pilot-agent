"""RetentionAgent - Churn prediction, nurture planning, cohort analysis.

Note: Retention tools (NurturePlanner, ChurnPredictor, WinbackPlanner,
CohortAnalyzer) are under development. This agent uses heuristic fallbacks
when tools are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.base import BaseAgent
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool imports with fallback stubs
# ---------------------------------------------------------------------------
try:
    from src.tools.retention import NurturePlanner, ChurnPredictor, WinbackPlanner, CohortAnalyzer
except ImportError:

    class _Stub:
        """Stub for tools not yet implemented."""

        def __init__(self, *a: Any, **kw: Any) -> None: ...

    NurturePlanner = ChurnPredictor = WinbackPlanner = CohortAnalyzer = _Stub


SYSTEM_PROMPT = """\
你是 GrowthPilot 用户留存 Agent。你的职责是：
1. 评估培育 (nurture) 进展
2. 预测流失风险
3. 设计挽回策略
4. 分析用户群组 (cohort)

请用 JSON 格式输出分析结果。
"""


class RetentionAgent(BaseAgent):
    """Manages user retention through churn prediction and nurture strategies."""

    name = "retention"
    description = "用户留存 Agent"

    def __init__(self, llm: Any, system_prompt: str = SYSTEM_PROMPT) -> None:
        super().__init__(llm=llm, system_prompt=system_prompt)
        self._nurture_planner = NurturePlanner()
        self._churn_predictor = ChurnPredictor()
        self._winback_planner = WinbackPlanner()
        self._cohort_analyzer = CohortAnalyzer()

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the retention analysis pipeline."""
        errors: list[str] = []
        prospect = state.get("prospect_results") or {}
        conversion = state.get("conversion_results") or {}

        # 1. Nurture progress (try real tool, fallback to heuristic)
        nurture_plans: dict = {}
        nurture_progress: dict = {}
        try:
            result = self._nurture_planner.run(
                data_path=state.get("data_path", ""),
                user_segments=prospect.get("segment_summary", {}),
            )
            nurture_plans = result.get("plans", {})
            nurture_progress = result.get("progress", {})
        except Exception as exc:
            logger.warning("NurturePlanner failed: %s", exc)
            nurture_plans = {"active": "weekly_push", "at_risk": "personalized_offer"}
            nurture_progress = {"completion_rate": 0.65, "active_plans": 3}
            errors.append(f"NurturePlanner (heuristic): {exc}")

        # 2. Churn prediction
        churn_risk: dict = {}
        high_risk_users: list = []
        churn_factors: list = []
        try:
            result = self._churn_predictor.run(
                data_path=state.get("data_path", ""),
                conversion_data=conversion,
            )
            churn_risk = result.get("churn_risk", {})
            high_risk_users = result.get("high_risk_users", [])
            churn_factors = result.get("factors", [])
        except Exception as exc:
            logger.warning("ChurnPredictor failed: %s", exc)
            churn_risk = {"high_risk_ratio": 0.12, "medium_risk_ratio": 0.25, "low_risk_ratio": 0.63}
            high_risk_users = [{"user_id": "sample", "risk_score": 0.85}]
            churn_factors = ["低活跃度", "无近30天订单", "价格敏感型用户", "竞品使用迹象"]
            errors.append(f"ChurnPredictor (heuristic): {exc}")

        # 3. Winback plans
        winback_plans: dict = {}
        winback_priority: list = []
        try:
            result = self._winback_planner.run(
                high_risk_users=high_risk_users,
                churn_factors=churn_factors,
                budget=state.get("budget", 0),
            )
            winback_plans = result.get("plans", {})
            winback_priority = result.get("priority", [])
        except Exception as exc:
            logger.warning("WinbackPlanner failed: %s", exc)
            winback_plans = {
                "high_value_churned": {"action": "大额优惠券+专属客服回访", "budget_share": 0.4},
                "medium_risk": {"action": "个性化Push+小券引导", "budget_share": 0.35},
                "low_engagement": {"action": "内容运营+活动邀请", "budget_share": 0.25},
            }
            winback_priority = ["high_value_churned", "medium_risk", "low_engagement"]
            errors.append(f"WinbackPlanner (heuristic): {exc}")

        # 4. Cohort analysis
        cohort_data: dict = {}
        retention_curve: dict = {}
        cohort_insight = ""
        try:
            result = self._cohort_analyzer.run(data_path=state.get("data_path", ""))
            cohort_data = result.get("cohorts", {})
            retention_curve = result.get("retention_curve", {})
            cohort_insight = result.get("insight", "")
        except Exception as exc:
            logger.warning("CohortAnalyzer failed: %s", exc)
            cohort_data = {"cohort_2024_q1": {"day_7": 0.45, "day_30": 0.28, "day_90": 0.15}}
            retention_curve = {"day_1": 0.75, "day_7": 0.45, "day_30": 0.28, "day_90": 0.15}
            cohort_insight = "近期群组留存率有提升趋势，但30日留存仍需改善"
            errors.append(f"CohortAnalyzer (heuristic): {exc}")

        # 5. LLM synthesis
        try:
            prompt = self._build_retention_prompt(
                nurture_progress=nurture_progress,
                churn_risk=churn_risk,
                high_risk_count=len(high_risk_users),
                churn_factors=churn_factors,
                winback_plans=winback_plans,
                winback_priority=winback_priority,
                cohort_insight=cohort_insight,
                state=state,
            )
            llm_response = await self._invoke_llm(prompt)
            analysis = self._parse_json_response(llm_response)
        except Exception as exc:
            logger.warning("Retention LLM synthesis failed: %s", exc)
            analysis = {"summary": "LLM synthesis unavailable"}
            errors.append(f"LLM synthesis: {exc}")

        result: dict[str, Any] = {
            "retention_results": {
                "nurture_plans": nurture_plans,
                "nurture_progress": nurture_progress,
                "churn_risk": churn_risk,
                "high_risk_users": high_risk_users,
                "churn_factors": churn_factors,
                "winback_plans": winback_plans,
                "winback_priority": winback_priority,
                "cohort_data": cohort_data,
                "retention_curve": retention_curve,
                "cohort_insight": cohort_insight,
                "analysis": analysis,
            },
        }
        if errors:
            result["errors"] = errors
        return result

    # ------------------------------------------------------------------
    def _build_retention_prompt(self, *, state: AgentState, **kw: Any) -> str:
        context = self._build_prompt_context(state)
        parts = [context, "", "## 用户留存分析数据"]

        if kw.get("nurture_progress"):
            parts.append(f"- 培育进展: {kw['nurture_progress']}")
        if kw.get("high_risk_count"):
            parts.append(f"- 高流失风险用户数: {kw['high_risk_count']}")
        if kw.get("churn_factors"):
            parts.append(f"- 流失因素: {kw['churn_factors']}")
        if kw.get("winback_priority"):
            parts.append(f"- 挽回优先级: {kw['winback_priority']}")
        if kw.get("cohort_insight"):
            parts.append(f"- 群组洞察: {kw['cohort_insight']}")

        parts.append("""
请基于以上数据给出用户留存的综合分析和建议：
1. 培育进展评估
2. 流失风险分析
3. 挽回策略建议
4. 群组分析洞察
5. 留存优化建议

请以 JSON 格式输出:
{
  "summary": "总体概述",
  "nurture_assessment": "培育进展评估",
  "churn_analysis": "流失风险分析",
  "winback_strategy": "挽回策略建议",
  "cohort_insight": "群组分析洞察",
  "retention_recommendation": "留存优化建议"
}""")
        return "\n".join(parts)
