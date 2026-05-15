"""Win-back campaign planner for churned users.

Generates segment-specific recall strategies based on churn reason
segmentation and historical win-back campaign performance data.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Default win-back strategy templates
# -----------------------------------------------------------------------

_DEFAULT_STRATEGIES: dict[str, dict[str, Any]] = {
    "price_sensitive": {
        "strategy": "price_discount",
        "channels": ["push", "sms"],
        "offer": "回归立享8折 + 无门槛10元券",
        "expected_winback_rate": 0.12,
        "budget_per_user": 15.0,
        "timing": "immediate",
    },
    "service_dissatisfied": {
        "strategy": "service_improvement",
        "channels": ["email", "in_app"],
        "offer": "为您升级服务体验 – 专属客服 + 服务保障",
        "expected_winback_rate": 0.08,
        "budget_per_user": 20.0,
        "timing": "3_days",
    },
    "competitor_switched": {
        "strategy": "competitive_offer",
        "channels": ["push", "sms", "email"],
        "offer": "限时回归礼：运费直减20%",
        "expected_winback_rate": 0.10,
        "budget_per_user": 25.0,
        "timing": "7_days",
    },
    "no_need": {
        "strategy": "need_stimulation",
        "channels": ["push"],
        "offer": "新功能上线：大件货运一键下单",
        "expected_winback_rate": 0.05,
        "budget_per_user": 5.0,
        "timing": "14_days",
    },
    "seasonal": {
        "strategy": "seasonal_recall",
        "channels": ["push", "email"],
        "offer": "换季搬家季 – 限时优惠回归",
        "expected_winback_rate": 0.15,
        "budget_per_user": 12.0,
        "timing": "seasonal",
    },
}


@ToolRegistry.register("winback_planner")
class WinbackPlanner:
    """Generate per-segment win-back plans for churned users.

    Uses churn segmentation (from :class:`ChurnPredictor`) and historical
    win-back performance data to produce optimal recall strategies.
    """

    def __init__(
        self,
        strategies: dict[str, dict[str, Any]] | None = None,
        min_segment_size: int = 50,
    ) -> None:
        self.strategies = strategies or _DEFAULT_STRATEGIES
        self.min_segment_size = min_segment_size

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_winback_plan(
        self,
        churn_segments: dict[str, Any],
        historical_winback_data: pd.DataFrame | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate win-back plans for each churn segment.

        Parameters
        ----------
        churn_segments : dict
            Output of :meth:`ChurnPredictor.segment_churned_users`,
            or a custom dict with segment names as keys and user
            DataFrames as values.
        historical_winback_data : optional
            Historical campaign results for calibration. Expected columns:
            ``segment``, ``channel``, ``winback_rate``, ``cost_per_winback``,
            ``roi``.

        Returns
        -------
        dict
            Per-segment strategies with:
                ``strategy``, ``channels``, ``offer``, ``expected_winback_rate``,
                ``estimated_cost``, ``estimated_roi``, ``timing``, ``user_count``
        """
        plan: dict[str, Any] = {}

        # Load historical performance for calibration
        hist_perf = self._load_historical_performance(historical_winback_data)

        for segment_name, segment_info in churn_segments.items():
            if segment_name in ("summary", "meta"):
                continue

            # Determine segment size
            if isinstance(segment_info, dict):
                user_count = segment_info.get("count", 0)
            elif isinstance(segment_info, pd.DataFrame):
                user_count = len(segment_info)
            else:
                user_count = 0

            if user_count < self.min_segment_size:
                plan[segment_name] = {
                    "strategy": "skip",
                    "reason": f"Segment too small ({user_count} < {self.min_segment_size})",
                    "user_count": user_count,
                }
                continue

            # Select and calibrate strategy
            strategy = self._select_strategy(segment_name, hist_perf)
            strategy["user_count"] = user_count

            # Estimate costs and ROI
            budget_per_user = strategy.get("budget_per_user", 10.0)
            expected_rate = strategy.get("expected_winback_rate", 0.05)

            # Calibrate expected rate with historical data
            if segment_name in hist_perf:
                hist_rate = hist_perf[segment_name].get("avg_winback_rate")
                if hist_rate is not None and hist_rate > 0:
                    # Blend: 60% historical, 40% template
                    expected_rate = 0.6 * hist_rate + 0.4 * expected_rate
                    strategy["expected_winback_rate"] = round(expected_rate, 4)

            strategy["estimated_cost"] = round(user_count * budget_per_user, 2)
            strategy["estimated_winbacks"] = int(user_count * expected_rate)
            strategy["estimated_roi"] = round(
                (user_count * expected_rate * 200 - user_count * budget_per_user)
                / max(user_count * budget_per_user, 1),
                2,
            )

            plan[segment_name] = strategy

        # Overall plan summary
        total_cost = sum(
            p.get("estimated_cost", 0) for p in plan.values() if isinstance(p, dict)
        )
        total_winbacks = sum(
            p.get("estimated_winbacks", 0) for p in plan.values() if isinstance(p, dict)
        )

        plan["summary"] = {
            "total_segments": len([k for k in plan if k != "summary"]),
            "total_estimated_cost": round(total_cost, 2),
            "total_estimated_winbacks": total_winbacks,
            "overall_winback_rate": round(
                total_winbacks / max(
                    sum(p.get("user_count", 0) for p in plan.values() if isinstance(p, dict)),
                    1,
                ),
                4,
            ),
            "priority_order": self._rank_segments(plan),
        }

        return plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_strategy(
        self,
        segment_name: str,
        hist_perf: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Select the best strategy for a segment."""
        # Direct match
        if segment_name in self.strategies:
            return dict(self.strategies[segment_name])

        # Map common segment naming patterns
        name_lower = segment_name.lower()
        mapping = {
            "high_risk": "price_sensitive",
            "medium_risk": "competitor_switched",
            "low_risk": "no_need",
            "price": "price_sensitive",
            "service": "service_dissatisfied",
            "competitor": "competitor_switched",
        }

        for key, mapped in mapping.items():
            if key in name_lower:
                return dict(self.strategies.get(mapped, self.strategies["price_sensitive"]))

        # Default: price_sensitive strategy
        return dict(self.strategies.get("price_sensitive", {
            "strategy": "general_discount",
            "channels": ["push"],
            "offer": "欢迎回来 – 专属优惠",
            "expected_winback_rate": 0.05,
            "budget_per_user": 10.0,
            "timing": "immediate",
        }))

    @staticmethod
    def _load_historical_performance(
        historical_winback_data: pd.DataFrame | dict[str, Any] | None,
    ) -> dict[str, dict[str, float]]:
        """Parse historical win-back data into a lookup dict."""
        result: dict[str, dict[str, float]] = {}

        if historical_winback_data is None:
            return result

        if isinstance(historical_winback_data, dict):
            return historical_winback_data

        if isinstance(historical_winback_data, pd.DataFrame):
            required = {"segment", "winback_rate"}
            if not required.issubset(historical_winback_data.columns):
                return result

            for segment, grp in historical_winback_data.groupby("segment"):
                result[str(segment)] = {
                    "avg_winback_rate": float(grp["winback_rate"].mean()),
                    "avg_cost": float(grp.get("cost_per_winback", pd.Series([10.0])).mean()),
                    "avg_roi": float(grp.get("roi", pd.Series([1.0])).mean()),
                    "campaigns": int(len(grp)),
                }

        return result

    @staticmethod
    def _rank_segments(plan: dict[str, Any]) -> list[str]:
        """Rank segments by expected ROI (descending)."""
        scored: list[tuple[str, float]] = []
        for seg_name, seg_plan in plan.items():
            if seg_name == "summary" or not isinstance(seg_plan, dict):
                continue
            roi = seg_plan.get("estimated_roi", 0.0)
            if seg_plan.get("strategy") == "skip":
                roi = -999
            scored.append((seg_name, roi))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored]
