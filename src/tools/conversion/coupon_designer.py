"""CouponDesigner - design discount vs threshold coupons.

Decides coupon type (折扣券/满减券), amount, threshold, and estimates
claim and redemption rates based on historical data.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Coupon type constants
COUPON_TYPE_DISCOUNT = "折扣券"     # percentage discount
COUPON_TYPE_THRESHOLD = "满减券"    # fixed amount off with minimum spend

# Default segment parameters (can be overridden)
SEGMENT_DEFAULTS: dict[str, dict[str, Any]] = {
    "new_user": {
        "preferred_type": COUPON_TYPE_THRESHOLD,
        "typical_order_value": 50.0,
        "price_sensitivity": 0.8,
        "coupon_affinity": 0.9,
    },
    "active": {
        "preferred_type": COUPON_TYPE_DISCOUNT,
        "typical_order_value": 80.0,
        "price_sensitivity": 0.5,
        "coupon_affinity": 0.6,
    },
    "moderate": {
        "preferred_type": COUPON_TYPE_THRESHOLD,
        "typical_order_value": 60.0,
        "price_sensitivity": 0.7,
        "coupon_affinity": 0.75,
    },
    "dormant": {
        "preferred_type": COUPON_TYPE_THRESHOLD,
        "typical_order_value": 55.0,
        "price_sensitivity": 0.85,
        "coupon_affinity": 0.8,
    },
    "high_value": {
        "preferred_type": COUPON_TYPE_DISCOUNT,
        "typical_order_value": 150.0,
        "price_sensitivity": 0.3,
        "coupon_affinity": 0.4,
    },
    "at_risk": {
        "preferred_type": COUPON_TYPE_THRESHOLD,
        "typical_order_value": 65.0,
        "price_sensitivity": 0.75,
        "coupon_affinity": 0.7,
    },
}


@ToolRegistry.register("coupon_designer")
class CouponDesigner:
    """Design discount and threshold coupons optimized for each segment."""

    def design_coupon(
        self,
        user_segment: str,
        historical_coupon_data: pd.DataFrame | dict[str, Any] | None = None,
        budget_constraint: float | None = None,
    ) -> dict[str, Any]:
        """Design an optimal coupon for a given user segment.

        Args:
            user_segment: segment name (e.g. "new_user", "dormant").
            historical_coupon_data: optional DataFrame or dict with past coupon
                performance. Expected columns: [coupon_type, amount, threshold,
                claim_rate, redemption_rate, orders_generated].
            budget_constraint: maximum average coupon cost per user.

        Returns:
            Dict with coupon_type, amount, threshold, and estimated rates.
        """
        seg_defaults = SEGMENT_DEFAULTS.get(
            user_segment, SEGMENT_DEFAULTS["moderate"]
        )

        # Analyse historical data if provided
        hist_analysis = self._analyse_history(historical_coupon_data, user_segment)

        # Determine coupon type
        if hist_analysis and hist_analysis.get("best_type"):
            coupon_type = hist_analysis["best_type"]
        else:
            coupon_type = seg_defaults["preferred_type"]

        aov = seg_defaults["typical_order_value"]
        sensitivity = seg_defaults["price_sensitivity"]
        affinity = seg_defaults["coupon_affinity"]

        # Design coupon parameters based on type
        if coupon_type == COUPON_TYPE_THRESHOLD:
            coupon = self._design_threshold_coupon(
                aov, sensitivity, affinity, budget_constraint, hist_analysis
            )
        else:
            coupon = self._design_discount_coupon(
                aov, sensitivity, affinity, budget_constraint, hist_analysis
            )

        coupon["segment"] = user_segment
        coupon["coupon_type"] = coupon_type

        # Estimate claim and redemption rates
        claim_rate = self._estimate_claim_rate(
            coupon, affinity, hist_analysis
        )
        redemption_rate = self._estimate_redemption_rate(
            coupon, sensitivity, hist_analysis
        )

        coupon["estimated_claim_rate"] = round(claim_rate, 4)
        coupon["estimated_redemption_rate"] = round(redemption_rate, 4)
        coupon["estimated_roi"] = round(
            self._estimate_roi(coupon, aov, claim_rate, redemption_rate), 4
        )

        return coupon

    def compare_coupon_types(
        self,
        segment: str,
        discount_history: pd.DataFrame | dict[str, Any] | None = None,
        threshold_history: pd.DataFrame | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compare discount vs threshold coupons for a segment.

        Args:
            segment: segment name.
            discount_history: historical performance of discount coupons.
            threshold_history: historical performance of threshold coupons.

        Returns:
            Comparison dict with recommendation.
        """
        seg_defaults = SEGMENT_DEFAULTS.get(segment, SEGMENT_DEFAULTS["moderate"])

        discount_metrics = self._summarise_history(discount_history, COUPON_TYPE_DISCOUNT)
        threshold_metrics = self._summarise_history(threshold_history, COUPON_TYPE_THRESHOLD)

        # If no history, use heuristic scoring
        if not discount_metrics["has_data"]:
            discount_metrics["avg_claim_rate"] = 0.35 * seg_defaults["coupon_affinity"]
            discount_metrics["avg_redemption_rate"] = 0.15 * seg_defaults["price_sensitivity"]
            discount_metrics["avg_roi"] = discount_metrics["avg_redemption_rate"] * 3.0

        if not threshold_metrics["has_data"]:
            threshold_metrics["avg_claim_rate"] = 0.55 * seg_defaults["coupon_affinity"]
            threshold_metrics["avg_redemption_rate"] = 0.25 * seg_defaults["price_sensitivity"]
            threshold_metrics["avg_roi"] = threshold_metrics["avg_redemption_rate"] * 2.5

        # Score each type
        discount_score = (
            discount_metrics["avg_claim_rate"] * 0.3
            + discount_metrics["avg_redemption_rate"] * 0.5
            + min(discount_metrics["avg_roi"] / 10.0, 0.2)
        )
        threshold_score = (
            threshold_metrics["avg_claim_rate"] * 0.3
            + threshold_metrics["avg_redemption_rate"] * 0.5
            + min(threshold_metrics["avg_roi"] / 10.0, 0.2)
        )

        recommendation = (
            COUPON_TYPE_DISCOUNT if discount_score >= threshold_score else COUPON_TYPE_THRESHOLD
        )

        return {
            "segment": segment,
            "discount_coupon": discount_metrics,
            "threshold_coupon": threshold_metrics,
            "discount_score": round(discount_score, 4),
            "threshold_score": round(threshold_score, 4),
            "recommendation": recommendation,
            "reason": (
                f"折扣券得分 {discount_score:.4f} vs 满减券得分 {threshold_score:.4f}，"
                f"推荐 {recommendation}"
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyse_history(
        self, data: Any, segment: str
    ) -> dict[str, Any] | None:
        """Analyse historical coupon performance."""
        if data is None:
            return None

        df = self._to_dataframe(data)
        if df is None or df.empty:
            return None

        result: dict[str, Any] = {"has_data": True}

        # Find the best performing coupon type
        if "redemption_rate" in df.columns and "coupon_type" in df.columns:
            by_type = df.groupby("coupon_type")["redemption_rate"].mean()
            if not by_type.empty:
                result["best_type"] = str(by_type.idxmax())
                result["best_redemption_rate"] = float(by_type.max())

        if "claim_rate" in df.columns:
            result["avg_claim_rate"] = float(df["claim_rate"].mean())

        if "redemption_rate" in df.columns:
            result["avg_redemption_rate"] = float(df["redemption_rate"].mean())

        if "amount" in df.columns:
            result["avg_amount"] = float(df["amount"].mean())

        return result

    def _summarise_history(
        self, data: Any, coupon_type: str
    ) -> dict[str, Any]:
        """Summarise historical data for a specific coupon type."""
        result: dict[str, Any] = {
            "has_data": False,
            "avg_claim_rate": 0.0,
            "avg_redemption_rate": 0.0,
            "avg_roi": 0.0,
            "sample_size": 0,
        }

        df = self._to_dataframe(data)
        if df is None or df.empty:
            return result

        result["has_data"] = True
        result["sample_size"] = len(df)

        if "claim_rate" in df.columns:
            result["avg_claim_rate"] = float(df["claim_rate"].mean())
        if "redemption_rate" in df.columns:
            result["avg_redemption_rate"] = float(df["redemption_rate"].mean())
        if "roi" in df.columns:
            result["avg_roi"] = float(df["roi"].mean())

        return result

    def _design_threshold_coupon(
        self,
        aov: float,
        sensitivity: float,
        affinity: float,
        budget: float | None,
        hist: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Design a 满减券 (threshold coupon)."""
        # Threshold typically 80-120% of AOV
        threshold = round(aov * (0.8 + sensitivity * 0.4), 0)
        # Discount amount: typically 15-30% of threshold
        base_amount = threshold * (0.15 + sensitivity * 0.15)

        if budget is not None:
            max_amount = budget
            base_amount = min(base_amount, max_amount)

        amount = max(5.0, round(base_amount / 5.0) * 5.0)  # round to 5

        return {
            "threshold": int(threshold),
            "amount": amount,
            "discount_depth": round(amount / threshold, 4),
        }

    def _design_discount_coupon(
        self,
        aov: float,
        sensitivity: float,
        affinity: float,
        budget: float | None,
        hist: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Design a 折扣券 (percentage discount coupon)."""
        # Discount rate typically 8.5-9.5折 (5%-15% off)
        base_discount = 0.95 - sensitivity * 0.10  # higher sensitivity -> deeper discount
        base_discount = max(0.85, min(0.95, base_discount))

        # Convert to Chinese discount notation (e.g. 9.5折)
        discount_zhe = round(base_discount * 10, 1)
        max_discount = round(aov * (1 - base_discount), 2)

        if budget is not None:
            if max_discount > budget:
                max_discount = budget
                discount_zhe = round((1 - budget / aov) * 10, 1)
                discount_zhe = max(8.0, discount_zhe)

        return {
            "discount_rate": f"{discount_zhe}折",
            "discount_pct": round((1 - base_discount) * 100, 1),
            "max_discount_amount": max_discount,
        }

    def _estimate_claim_rate(
        self,
        coupon: dict[str, Any],
        affinity: float,
        hist: dict[str, Any] | None,
    ) -> float:
        """Estimate coupon claim rate."""
        if hist and "avg_claim_rate" in hist:
            base = hist["avg_claim_rate"]
        else:
            base = 0.5 * affinity

        # Boost for higher discount depth
        if "discount_depth" in coupon:
            depth = coupon["discount_depth"]
            base *= 1.0 + depth * 2.0
        elif "discount_pct" in coupon:
            pct = coupon["discount_pct"] / 100.0
            base *= 1.0 + pct * 3.0

        return min(base, 0.95)

    def _estimate_redemption_rate(
        self,
        coupon: dict[str, Any],
        sensitivity: float,
        hist: dict[str, Any] | None,
    ) -> float:
        """Estimate coupon redemption rate (of claimed coupons)."""
        if hist and "avg_redemption_rate" in hist:
            base = hist["avg_redemption_rate"]
        else:
            base = 0.2 * sensitivity

        # Threshold coupons have higher redemption if threshold is close to natural spend
        if "threshold" in coupon and "amount" in coupon:
            ratio = coupon["amount"] / coupon["threshold"]
            base *= 1.0 + ratio * 3.0

        return min(base, 0.8)

    def _estimate_roi(
        self,
        coupon: dict[str, Any],
        aov: float,
        claim_rate: float,
        redemption_rate: float,
    ) -> float:
        """Estimate ROI = incremental revenue / coupon cost."""
        # Revenue per redeemed coupon
        revenue = aov * redemption_rate * claim_rate

        # Cost per coupon
        if "amount" in coupon:
            cost = coupon["amount"] * claim_rate * redemption_rate
        elif "max_discount_amount" in coupon:
            cost = coupon["max_discount_amount"] * claim_rate * redemption_rate
        else:
            cost = 1.0

        if cost <= 0:
            return 0.0
        return revenue / cost

    @staticmethod
    def _to_dataframe(data: Any) -> pd.DataFrame | None:
        """Convert various inputs to a DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            try:
                return pd.DataFrame(data)
            except Exception:
                return None
        if isinstance(data, list):
            try:
                return pd.DataFrame(data)
            except Exception:
                return None
        return None
