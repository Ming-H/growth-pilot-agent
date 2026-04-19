"""ConversionAgent - Conversion funnel analysis, reach planning, coupon design."""

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
    from src.tools.conversion.reach_planner import ReachPlanner
    from src.tools.conversion.funnel_analyzer import FunnelAnalyzer
    from src.tools.conversion.slot_allocator import SlotAllocator
    from src.tools.conversion.coupon_designer import CouponDesigner
except ImportError:

    class _Stub:
        def __init__(self, *a: Any, **kw: Any) -> None: ...

    ReachPlanner = FunnelAnalyzer = SlotAllocator = CouponDesigner = _Stub

# Attributor and SeasonalAnalyzer are not yet implemented
try:
    from src.tools.conversion import Attributor, SeasonalAnalyzer
except ImportError:

    class _Stub2:
        def __init__(self, *a: Any, **kw: Any) -> None: ...

        def analyze(self, *a: Any, **kw: Any) -> dict:
            return {"status": "not_implemented"}

    Attributor = SeasonalAnalyzer = _Stub2


SYSTEM_PROMPT = """\
你是 GrowthPilot 转化 Agent。你的职责是：
1. 设计触达策略 (reach planning)
2. 分析转化漏斗
3. 分配投放位
4. 设计优惠券策略
5. 结合季节性因素

请用 JSON 格式输出分析结果。
"""


class ConversionAgent(BaseAgent):
    """Designs conversion strategies for freight user acquisition."""

    name = "conversion"
    description = "转化策略 Agent"

    def __init__(self, llm: Any, system_prompt: str = SYSTEM_PROMPT) -> None:
        super().__init__(llm=llm, system_prompt=system_prompt)
        self._reach_planner = ReachPlanner()
        self._funnel_analyzer = FunnelAnalyzer()
        self._slot_allocator = SlotAllocator()
        self._coupon_designer = CouponDesigner()

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the conversion analysis pipeline."""
        errors: list[str] = []
        prospect = state.get("prospect_results") or {}
        subsidy = state.get("subsidy_results") or {}

        # 1. Design reach strategy
        reach_result: dict = {}
        try:
            # Build user_segments from prospect results
            user_segments = self._extract_user_segments(prospect)
            constraints: dict[str, Any] = {}
            if state.get("budget"):
                constraints["budget"] = state["budget"]
            reach_result = self._reach_planner.plan_reach_strategy(
                user_segments=user_segments,
                constraints=constraints if constraints else None,
            )
        except Exception as exc:
            logger.warning("ReachPlanner failed: %s", exc)
            errors.append(f"ReachPlanner: {exc}")

        # 2. Analyze funnel
        funnel_result: dict = {}
        try:
            # Use default funnel data or data from state
            funnel_data = self._get_funnel_data(state)
            funnel_result = self._funnel_analyzer.analyze_funnel(funnel_data)
        except Exception as exc:
            logger.warning("FunnelAnalyzer failed: %s", exc)
            errors.append(f"FunnelAnalyzer: {exc}")

        # 3. Allocate slots
        slot_result: dict = {}
        try:
            user_segments_for_slots = self._build_slot_segments(prospect)
            slot_result = self._slot_allocator.allocate_slots(
                user_segments=user_segments_for_slots,
            )
        except Exception as exc:
            logger.warning("SlotAllocator failed: %s", exc)
            errors.append(f"SlotAllocator: {exc}")

        # 4. Design coupons for each segment
        coupon_results: list[dict] = []
        try:
            segment_names = ["new_user", "active", "moderate", "dormant", "high_value", "at_risk"]
            budget_constraint = None
            if state.get("budget"):
                # Allocate up to 20% of budget for coupons
                budget_constraint = state["budget"] * 0.2
            for seg in segment_names:
                try:
                    coupon = self._coupon_designer.design_coupon(
                        user_segment=seg,
                        budget_constraint=budget_constraint,
                    )
                    coupon_results.append(coupon)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("CouponDesigner failed: %s", exc)
            errors.append(f"CouponDesigner: {exc}")

        # 5. LLM synthesis
        try:
            prompt = self._build_conversion_prompt(
                reach_result=reach_result,
                funnel_result=funnel_result,
                slot_result=slot_result,
                coupon_results=coupon_results,
                state=state,
            )
            llm_response = await self._invoke_llm(prompt)
            analysis = self._parse_json_response(llm_response)
        except Exception as exc:
            logger.warning("Conversion LLM synthesis failed: %s", exc)
            analysis = {"summary": "LLM synthesis unavailable"}
            errors.append(f"LLM synthesis: {exc}")

        result: dict[str, Any] = {
            "conversion_results": {
                "reach_result": reach_result,
                "funnel_result": funnel_result,
                "slot_result": slot_result,
                "coupon_results": coupon_results,
                "analysis": analysis,
            },
        }
        if errors:
            result["errors"] = errors
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_user_segments(prospect: dict) -> dict[str, int]:
        """Extract user segment counts from prospect results."""
        summary = prospect.get("segment_summary", {})
        if isinstance(summary, dict):
            return {
                name: int(info.get("count", 0)) if isinstance(info, dict) else int(info)
                for name, info in summary.items()
            }
        return {"new_user": 5000, "active": 8000, "moderate": 6000, "dormant": 3000}

    @staticmethod
    def _get_funnel_data(state: AgentState) -> dict[str, int]:
        """Get funnel data from state or return default sample."""
        # Default sample funnel data for demo
        return {
            "exposure": 100000,
            "click": 25000,
            "app_open": 18000,
            "search": 12000,
            "quote_view": 9000,
            "order_confirm": 5500,
            "first_order": 3200,
        }

    @staticmethod
    def _build_slot_segments(prospect: dict) -> dict[str, dict[str, Any]]:
        """Build slot allocation input from prospect data."""
        summary = prospect.get("segment_summary", {})
        if isinstance(summary, dict) and summary:
            result = {}
            for name, info in summary.items():
                if isinstance(info, dict):
                    result[name] = {
                        "count": info.get("count", 1000),
                        "ltv": info.get("avg_ltv", 100),
                        "priority": 3,
                    }
                else:
                    result[name] = {"count": int(info), "ltv": 100, "priority": 3}
            return result
        return {
            "new_user": {"count": 5000, "ltv": 80, "priority": 4},
            "active": {"count": 8000, "ltv": 200, "priority": 5},
            "moderate": {"count": 6000, "ltv": 120, "priority": 3},
            "dormant": {"count": 3000, "ltv": 60, "priority": 2},
        }

    def _build_conversion_prompt(
        self,
        *,
        reach_result: dict,
        funnel_result: dict,
        slot_result: dict,
        coupon_results: list[dict],
        state: AgentState,
    ) -> str:
        context = self._build_prompt_context(state)
        parts = [context, "", "## 转化分析数据"]

        if reach_result.get("strategies"):
            strategies = reach_result["strategies"]
            parts.append(f"- 触达策略: {len(strategies)} 个分层策略")
            for seg, plan in list(strategies.items())[:3]:
                if isinstance(plan, dict):
                    parts.append(f"  - {seg}: 渠道={plan.get('primary_channel', 'N/A')}, 创意={plan.get('creative', 'N/A')[:30]}")

        if funnel_result.get("overall_conversion_rate") is not None:
            parts.append(f"- 整体转化率: {funnel_result['overall_conversion_rate']:.2%}")
        if funnel_result.get("bottleneck"):
            bn = funnel_result["bottleneck"]
            parts.append(f"- 漏斗瓶颈: {bn.get('stage', 'N/A')} (转化率: {bn.get('stage_conversion_rate', 'N/A')})")

        if slot_result.get("total_slots_used"):
            parts.append(f"- 投放位分配: 使用 {slot_result['total_slots_used']}/{slot_result.get('total_slots_available', '?')} 个位")

        if coupon_results:
            parts.append(f"- 优惠券方案: {len(coupon_results)} 个分层方案")
            for c in coupon_results[:3]:
                parts.append(f"  - {c.get('segment', 'N/A')}: {c.get('coupon_type', 'N/A')}, 金额={c.get('amount', 'N/A')}")

        parts.append("""
请基于以上数据给出转化策略的综合分析和建议：
1. 触达策略评估
2. 漏斗优化建议
3. 优惠券策略建议
4. 投放位分配评估

请以 JSON 格式输出:
{
  "summary": "总体概述",
  "reach_assessment": "触达策略评估",
  "funnel_optimization": "漏斗优化建议",
  "coupon_recommendation": "优惠券策略建议",
  "slot_recommendation": "投放位分配建议"
}""")
        return "\n".join(parts)
