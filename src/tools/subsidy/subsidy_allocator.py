"""SubsidyAllocator - generate final subsidy plan.

Combines causal inference results, elasticity estimates, and budget
allocation to produce a final per-segment subsidy plan with coupon details.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default coupon templates per segment
DEFAULT_COUPON_CONFIG: dict[str, dict[str, Any]] = {
    "new_user": {"type": "满减券", "default_threshold": 30, "max_amount": 25},
    "active": {"type": "折扣券", "default_threshold": 0, "max_amount": 15},
    "moderate": {"type": "满减券", "default_threshold": 40, "max_amount": 20},
    "dormant": {"type": "满减券", "default_threshold": 20, "max_amount": 30},
    "at_risk": {"type": "满减券", "default_threshold": 35, "max_amount": 20},
    "high_value": {"type": "折扣券", "default_threshold": 0, "max_amount": 10},
}


@ToolRegistry.register("subsidy_allocator")
class SubsidyAllocator:
    """Generate final subsidy allocation plan from analysis results."""

    def allocate(
        self,
        causal_results: dict[str, Any],
        elasticity_results: dict[str, Any],
        budget_plan: dict[str, Any],
        user_segments: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Generate final per-segment subsidy plan.

        Args:
            causal_results: output from CausalInferenceEngine, with per-segment ATE.
                Expected structure: {"ate": float, ...} or {"cate": {seg: {ate, ...}}}.
            elasticity_results: output from ElasticityEstimator.
                Expected structure: {"segment_elasticities": [{segment, elasticity, ...}]}.
            budget_plan: output from BudgetOptimizer.
                Expected structure: {"allocation": {seg: {coupon_amount, ...}}}.
            user_segments: optional mapping of segment -> user_count. If not
                provided, extracted from budget_plan.

        Returns:
            Final subsidy plan with per-segment coupon details, thresholds,
            expected outcomes, and execution guidance.
        """
        if "error" in budget_plan:
            return {"error": f"budget_plan has error: {budget_plan['error']}"}

        allocation = budget_plan.get("allocation", {})
        if not allocation:
            return {"error": "budget_plan contains no allocation"}

        # Extract per-segment ATE
        ate_by_segment = self._extract_ate(causal_results)

        # Extract per-segment elasticity
        elasticity_by_segment = self._extract_elasticity(elasticity_results)

        # Build final plan
        plan_segments: list[dict[str, Any]] = []
        total_cost = 0.0
        total_expected_orders = 0.0
        total_expected_revenue = 0.0

        for seg_name, seg_allocation in allocation.items():
            coupon_amount = seg_allocation.get("coupon_amount", 0)
            user_count = seg_allocation.get("user_count", 0)
            expected_inc = seg_allocation.get("expected_incremental_orders", 0)

            ate = ate_by_segment.get(seg_name, 0.01)
            elasticity = elasticity_by_segment.get(seg_name, -1.0)

            # Design coupon details
            coupon_config = DEFAULT_COUPON_CONFIG.get(
                seg_name, DEFAULT_COUPON_CONFIG.get("moderate", {})
            )
            coupon_detail = self._design_coupon_detail(
                seg_name, coupon_amount, elasticity, coupon_config
            )

            # Estimate expected revenue (assume average order value ~ 80 yuan)
            aov_estimate = self._estimate_aov(seg_name)
            expected_revenue = expected_inc * aov_estimate

            seg_cost = user_count * coupon_amount
            total_cost += seg_cost
            total_expected_orders += expected_inc
            total_expected_revenue += expected_revenue

            plan_segments.append(
                {
                    "segment": seg_name,
                    "user_count": user_count,
                    "coupon": coupon_detail,
                    "causal_effect": {
                        "ate": round(ate, 6),
                        "elasticity": round(elasticity, 4),
                    },
                    "expected_outcomes": {
                        "incremental_orders": round(expected_inc, 2),
                        "expected_revenue": round(expected_revenue, 2),
                        "cost": round(seg_cost, 2),
                        "roi": round(expected_revenue / seg_cost, 4) if seg_cost > 0 else 0.0,
                    },
                    "execution_priority": self._compute_priority(seg_name, ate, elasticity, expected_inc),
                }
            )

        # Sort by execution priority (highest first)
        plan_segments.sort(key=lambda x: x["execution_priority"]["score"], reverse=True)

        total_budget = budget_plan.get("total_budget", total_cost)
        roi = total_expected_revenue / total_cost if total_cost > 0 else 0.0

        return {
            "plan": plan_segments,
            "summary": {
                "total_segments": len(plan_segments),
                "total_users_targeted": sum(s["user_count"] for s in plan_segments),
                "total_subsidy_cost": round(total_cost, 2),
                "total_budget": round(total_budget, 2),
                "budget_utilization": round(total_cost / total_budget, 4) if total_budget > 0 else 0.0,
                "expected_incremental_orders": round(total_expected_orders, 2),
                "expected_incremental_revenue": round(total_expected_revenue, 2),
                "overall_roi": round(roi, 4),
            },
            "execution_guidance": self._generate_execution_guidance(plan_segments),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ate(causal_results: dict[str, Any]) -> dict[str, float]:
        """Extract per-segment ATE from causal results."""
        ate_map: dict[str, float] = {}

        # Check if CATE results exist
        cate = causal_results.get("cate", {})
        if isinstance(cate, dict):
            for seg_key, seg_result in cate.items():
                if isinstance(seg_result, dict) and "ate" in seg_result:
                    ate_map[seg_key] = seg_result["ate"]

        # Check if it's a simple ATE result
        if not ate_map and "ate" in causal_results:
            ate_map["overall"] = causal_results["ate"]

        return ate_map

    @staticmethod
    def _extract_elasticity(elasticity_results: dict[str, Any]) -> dict[str, float]:
        """Extract per-segment elasticity from elasticity results."""
        elastic_map: dict[str, float] = {}

        seg_elasticities = elasticity_results.get("segment_elasticities", [])
        for item in seg_elasticities:
            seg = item.get("segment", "")
            e = item.get("elasticity")
            if seg and e is not None:
                elastic_map[seg] = e

        # Fallback to overall
        if not elastic_map:
            overall = elasticity_results.get("overall_elasticity", {})
            if isinstance(overall, dict) and "elasticity" in overall:
                elastic_map["overall"] = overall["elasticity"]

        return elastic_map

    @staticmethod
    def _design_coupon_detail(
        segment: str,
        amount: float,
        elasticity: float,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Design detailed coupon parameters for a segment."""
        coupon_type = config.get("type", "满减券")

        if coupon_type == "满减券":
            threshold = config.get("default_threshold", 30)
            # Adjust threshold based on coupon amount
            if amount > 15:
                threshold = max(threshold, int(amount * 2))
            else:
                threshold = max(threshold, int(amount * 1.5))

            return {
                "type": "满减券",
                "amount": round(amount, 2),
                "threshold": threshold,
                "display_text": f"满{threshold}减{int(amount)}",
                "validity_days": 7 if segment in ("new_user", "dormant") else 5,
            }
        else:
            # Discount coupon
            if amount > 0 and elasticity < -1.0:
                discount_pct = min(20, amount * 2)
            else:
                discount_pct = min(15, amount)

            zhe = round(10 - discount_pct / 10, 1)
            return {
                "type": "折扣券",
                "discount": f"{zhe}折",
                "discount_pct": round(discount_pct, 1),
                "max_discount_amount": round(amount, 2),
                "display_text": f"{zhe}折最高减{int(amount)}元",
                "validity_days": 5 if segment == "active" else 7,
            }

    @staticmethod
    def _estimate_aov(segment: str) -> float:
        """Estimate average order value per segment."""
        aov_map: dict[str, float] = {
            "new_user": 50.0,
            "active": 85.0,
            "moderate": 65.0,
            "dormant": 55.0,
            "at_risk": 70.0,
            "high_value": 150.0,
        }
        return aov_map.get(segment, 70.0)

    @staticmethod
    def _compute_priority(
        segment: str, ate: float, elasticity: float, expected_inc: float
    ) -> dict[str, Any]:
        """Compute execution priority for a segment."""
        # Score based on: ATE impact, elasticity (more elastic = higher priority),
        # and expected volume
        ate_score = min(abs(ate) * 100, 10)
        elastic_score = min(abs(elasticity) * 3, 5) if elasticity < 0 else 1.0
        volume_score = min(expected_inc / 100, 5)

        total_score = ate_score + elastic_score + volume_score

        if total_score >= 10:
            priority = "high"
        elif total_score >= 5:
            priority = "medium"
        else:
            priority = "low"

        return {
            "priority": priority,
            "score": round(total_score, 4),
            "rationale": f"ATE={ate:.4f}, elasticity={elasticity:.2f}, "
            f"expected_incr={expected_inc:.1f}",
        }

    @staticmethod
    def _generate_execution_guidance(plan_segments: list[dict[str, Any]]) -> list[str]:
        """Generate execution guidance based on the plan."""
        guidance: list[str] = []

        if not plan_segments:
            return ["No segments to allocate."]

        # Highest priority segment
        top = plan_segments[0]
        guidance.append(
            f"1. 优先执行 {top['segment']} 群体投放："
            f"预计带来 {top['expected_outcomes']['incremental_orders']:.0f} 增量订单"
        )

        # Segments with high elasticity
        elastic_segs = [
            s for s in plan_segments
            if s["causal_effect"]["elasticity"] < -1.0
        ]
        if elastic_segs:
            names = ", ".join(s["segment"] for s in elastic_segs)
            guidance.append(
                f"2. 高价格敏感群体 ({names})：建议搭配限时优惠文案提升紧迫感"
            )

        # Budget-constrained advice
        total_cost = sum(s["expected_outcomes"]["cost"] for s in plan_segments)
        guidance.append(
            f"3. 总补贴成本 {total_cost:.0f} 元，"
            f"预计 ROI {sum(s['expected_outcomes']['expected_revenue'] for s in plan_segments) / total_cost:.1f}x"
        )

        # Timing advice
        guidance.append(
            "4. 建议分批次执行：先投放高优先级群体，观察3天后调整低优先级群体策略"
        )

        return guidance
