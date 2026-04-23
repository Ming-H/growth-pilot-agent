"""ConversionExpert - Conversion funnel analysis, reach planning, coupon design."""

from __future__ import annotations

import logging
from typing import Any

from src.core.expert import ExpertAgentBase
from src.prompts.templates.agent_prompts import ConversionPrompt

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


class ConversionExpert(ExpertAgentBase):
    """Designs conversion strategies for freight user acquisition."""

    name = "conversion"
    description = "转化策略 Expert"

    # ------------------------------------------------------------------
    # Keyword matching
    # ------------------------------------------------------------------
    _KEYWORDS = ("转化", "漏斗", "优惠券", "转化率", "coupon", "funnel", "conversion")

    @classmethod
    def can_handle(cls, query: str) -> bool:
        """Return True if *query* mentions conversion-related topics."""
        lower = query.lower()
        return any(kw in lower for kw in cls._KEYWORDS)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_tools(self) -> dict[str, Any]:
        """Instantiate and return the four conversion tools."""
        return {
            "reach_planner": ReachPlanner(),
            "funnel_analyzer": FunnelAnalyzer(),
            "slot_allocator": SlotAllocator(),
            "coupon_designer": CouponDesigner(),
        }

    def _get_system_prompt(self) -> str:
        """Build the default system prompt from ConversionPrompt template."""
        template = ConversionPrompt()
        return f"{template.role_definition}\n\n{template.business_context}"

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def _execute_pipeline(self, params: dict) -> dict:
        """Run all four conversion tool steps and return raw results dict.

        Returns a dict with keys: reach_result, funnel_result,
        slot_result, coupon_results, errors.
        """
        tools = self._init_tools()

        prospect = params.get("prospect_results", {})
        budget = params.get("budget")

        errors: list[str] = []

        # 1. Design reach strategy
        reach_result: dict = {}
        try:
            user_segments = self._extract_user_segments(prospect)
            constraints: dict[str, Any] | None = None
            if budget is not None:
                constraints = {"budget": budget}
            reach_result = tools["reach_planner"].plan_reach_strategy(
                user_segments=user_segments,
                constraints=constraints,
            )
        except Exception as exc:
            logger.warning("ReachPlanner failed: %s", exc)
            errors.append(f"ReachPlanner: {exc}")

        # 2. Analyze funnel
        funnel_result: dict = {}
        try:
            funnel_data = self._get_funnel_data(params)
            funnel_result = tools["funnel_analyzer"].analyze_funnel(funnel_data)
        except Exception as exc:
            logger.warning("FunnelAnalyzer failed: %s", exc)
            errors.append(f"FunnelAnalyzer: {exc}")

        # 3. Allocate slots
        slot_result: dict = {}
        try:
            user_segments_for_slots = self._build_slot_segments(prospect)
            slot_result = tools["slot_allocator"].allocate_slots(
                user_segments=user_segments_for_slots,
            )
        except Exception as exc:
            logger.warning("SlotAllocator failed: %s", exc)
            errors.append(f"SlotAllocator: {exc}")

        # 4. Design coupons for each segment
        coupon_results: list[dict] = []
        try:
            segment_names = [
                "new_user", "active", "moderate", "dormant",
                "high_value", "at_risk",
            ]
            budget_constraint = None
            if budget is not None:
                budget_constraint = budget * 0.2
            for seg in segment_names:
                try:
                    coupon = tools["coupon_designer"].design_coupon(
                        user_segment=seg,
                        budget_constraint=budget_constraint,
                    )
                    coupon_results.append(coupon)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("CouponDesigner failed: %s", exc)
            errors.append(f"CouponDesigner: {exc}")

        return {
            "reach_result": reach_result,
            "funnel_result": funnel_result,
            "slot_result": slot_result,
            "coupon_results": coupon_results,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_synthesis_prompt(self, params: dict, results: dict) -> str:
        """Build the LLM synthesis prompt from tool results.

        Adapted from the former ``_build_conversion_prompt``.
        """
        reach_result = results.get("reach_result", {})
        funnel_result = results.get("funnel_result", {})
        slot_result = results.get("slot_result", {})
        coupon_results = results.get("coupon_results", [])

        # Build reach summary
        reach_summary = ""
        if reach_result.get("strategies"):
            strategies = reach_result["strategies"]
            parts = [f"{len(strategies)} 个分层策略"]
            for seg, plan in list(strategies.items())[:3]:
                if isinstance(plan, dict):
                    parts.append(
                        f"  - {seg}: 渠道={plan.get('primary_channel', 'N/A')}, "
                        f"创意={plan.get('creative', 'N/A')[:30]}"
                    )
            reach_summary = "\n".join(parts)

        # Build funnel metrics
        funnel_overall_cvr = ""
        if funnel_result.get("overall_conversion_rate") is not None:
            funnel_overall_cvr = f"{funnel_result['overall_conversion_rate']:.2%}"

        bottleneck_stage = ""
        bottleneck_cvr = ""
        if funnel_result.get("bottleneck"):
            bn = funnel_result["bottleneck"]
            bottleneck_stage = bn.get("stage", "N/A")
            bottleneck_cvr = str(bn.get("stage_conversion_rate", "N/A"))

        # Build slot usage
        slot_usage = ""
        if slot_result.get("total_slots_used"):
            slot_usage = (
                f"使用 {slot_result['total_slots_used']}/"
                f"{slot_result.get('total_slots_available', '?')} 个位"
            )

        # Build coupon summary
        coupon_summary = ""
        if coupon_results:
            parts = [f"{len(coupon_results)} 个分层方案"]
            for c in coupon_results[:3]:
                parts.append(
                    f"  - {c.get('segment', 'N/A')}: "
                    f"{c.get('coupon_type', 'N/A')}, "
                    f"金额={c.get('amount', 'N/A')}"
                )
            coupon_summary = "\n".join(parts)

        return ConversionPrompt().render(
            user_query=params.get("query", ""),
            season=params.get("season", "当前"),
            kpi_baseline=params.get("kpi_baseline", {}),
            memory_context=params.get("memory_context", {}),
            reach_summary=reach_summary,
            funnel_overall_cvr=funnel_overall_cvr,
            bottleneck_stage=bottleneck_stage,
            bottleneck_cvr=bottleneck_cvr,
            slot_usage=slot_usage,
            coupon_summary=coupon_summary,
        )

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
    def _get_funnel_data(params: dict) -> dict[str, int]:
        """Get funnel data from params or return default sample."""
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
