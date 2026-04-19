"""OCPX bid optimization with PID control and simulation.

Provides bid calculation, PID-based auto-adjustment, and Monte Carlo-style
bid simulation for budget planning.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class BidOptimizer:
    """OCPX (Optimized Cost Per X) bid optimizer.

    Supports eCPC bidding with PID controller feedback and
    bid-level simulation for budget allocation.
    """

    # ------------------------------------------------------------------
    # Core bidding
    # ------------------------------------------------------------------

    def ecpc_bid(
        self,
        current_bid: float,
        cvr: float,
        target_cpa: float,
        alpha: float = 0.5,
    ) -> float:
        """Calculate eCPC (effective cost-per-click) bid.

        Formula::

            bid = alpha * base_bid * cvr / target_cpa
                  + (1 - alpha) * current_bid

        The alpha parameter controls the blend between the formulaic bid
        and the current bid for smoother transitions.

        Args:
            current_bid: Current bid price (CNY).
            cvr: Estimated conversion rate (0.0 ~ 1.0).
            target_cpa: Target cost-per-acquisition (CNY).
            alpha: Blending factor between formulaic and current bid (0~1).

        Returns:
            Adjusted bid price (CNY), rounded to 2 decimal places.
        """
        if target_cpa <= 0:
            return 0.0
        if cvr <= 0:
            return 0.0
        if current_bid < 0:
            return 0.0

        # Theoretical bid: if we pay this per click, we expect target_cpa per conversion
        # bid_per_click = target_cpa * cvr  (inverse: bid = target_cpa * cvr)
        # But the requirement says: bid = base_bid * cvr / target_cpa
        formulaic_bid = current_bid * cvr / target_cpa

        # Blend with current bid for smooth transitions
        new_bid = alpha * formulaic_bid + (1 - alpha) * current_bid

        # Clamp to reasonable range (at least 0.01, at most 10x current bid)
        new_bid = max(0.01, min(new_bid, current_bid * 10.0))

        return round(float(new_bid), 2)

    # ------------------------------------------------------------------
    # PID controller
    # ------------------------------------------------------------------

    def pid_controller(
        self,
        error_history: list[float],
        kp: float = 0.3,
        ki: float = 0.1,
        kd: float = 0.05,
    ) -> float:
        """Compute a PID control signal from error history.

        Standard PID formula::

            output = Kp * e(t) + Ki * sum(errors) + Kd * (e(t) - e(t-1))

        Used to adjust bid multipliers based on CPA tracking error
        (actual CPA minus target CPA).

        Args:
            error_history: Chronological list of errors (most recent last).
                          Positive means overshoot (actual > target).
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.

        Returns:
            PID output value that can be used as a bid adjustment factor.
        """
        if not error_history:
            return 0.0

        # Proportional term
        p_term = kp * error_history[-1]

        # Integral term (sum of all errors, with anti-windup clamping)
        integral = float(np.sum(error_history))
        # Anti-windup: clamp integral to prevent accumulation blowup
        integral = max(-100.0, min(integral, 100.0))
        i_term = ki * integral

        # Derivative term
        if len(error_history) >= 2:
            derivative = error_history[-1] - error_history[-2]
        else:
            derivative = 0.0
        d_term = kd * derivative

        output = p_term + i_term + d_term
        return round(float(output), 4)

    # ------------------------------------------------------------------
    # Bid simulation
    # ------------------------------------------------------------------

    def bid_simulation(
        self,
        budget: float,
        base_bid: float,
        ctr_curve: dict[str, list[float]],
        cvr_curve: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Simulate expected outcomes at different bid levels.

        Args:
            budget: Total budget for the campaign (CNY).
            base_bid: Base bid price (CNY).
            ctr_curve: Dict with keys 'bid_multipliers' and 'ctr_values'.
                       bid_multipliers: e.g. [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
                       ctr_values: corresponding CTR at each multiplier level.
            cvr_curve: Dict with keys 'bid_multipliers' and 'cvr_values'.
                       cvr_values: corresponding CVR at each multiplier level.

        Returns:
            Dict with:
                - 'bid_levels': list of bid amounts tested
                - 'results': list of dicts per bid level with estimated metrics
                - 'optimal_bid': the bid level with best ROI
                - 'recommendation': summary dict
        """
        if budget <= 0 or base_bid <= 0:
            return {"bid_levels": [], "results": [], "optimal_bid": 0.0, "recommendation": {}}

        bid_multipliers = ctr_curve.get("bid_multipliers", [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
        ctr_values = ctr_curve.get("ctr_values", [0.01, 0.015, 0.02, 0.023, 0.025, 0.026])
        cvr_values = cvr_curve.get("cvr_values", [0.02, 0.03, 0.04, 0.045, 0.047, 0.048])

        if len(bid_multipliers) != len(ctr_values) or len(bid_multipliers) != len(cvr_values):
            return {
                "bid_levels": [],
                "results": [],
                "optimal_bid": 0.0,
                "recommendation": {"error": "Mismatched curve lengths"},
            }

        results: list[dict[str, Any]] = []
        best_roi = -1.0
        optimal_bid = base_bid

        for mult, ctr, cvr in zip(bid_multipliers, ctr_values, cvr_values):
            bid_amount = base_bid * mult
            impressions = int(budget / bid_amount) if bid_amount > 0 else 0
            clicks = int(impressions * ctr)
            conversions = int(clicks * cvr)
            spend = clicks * bid_amount
            cpa = spend / conversions if conversions > 0 else float("inf")
            revenue = conversions * 80.0  # assumed average revenue per conversion
            roi = revenue / spend if spend > 0 else 0.0

            result = {
                "bid_multiplier": round(mult, 2),
                "bid_amount": round(bid_amount, 2),
                "estimated_impressions": impressions,
                "estimated_clicks": clicks,
                "estimated_conversions": conversions,
                "estimated_spend": round(spend, 2),
                "estimated_cpa": round(cpa, 2) if cpa != float("inf") else None,
                "estimated_revenue": round(revenue, 2),
                "roi": round(roi, 4),
            }
            results.append(result)

            if roi > best_roi:
                best_roi = roi
                optimal_bid = bid_amount

        # Recommendation
        if results:
            # Also find the result with most conversions while staying within budget
            feasible = [r for r in results if r["estimated_spend"] <= budget * 1.05]
            if feasible:
                best_conv = max(feasible, key=lambda r: r["estimated_conversions"])
                recommendation = {
                    "optimal_bid_for_roi": round(optimal_bid, 2),
                    "optimal_bid_for_volume": round(best_conv["bid_amount"], 2),
                    "best_roi": round(best_roi, 4),
                    "max_conversions": best_conv["estimated_conversions"],
                    "max_conversion_cpa": best_conv["estimated_cpa"],
                }
            else:
                recommendation = {
                    "optimal_bid_for_roi": round(optimal_bid, 2),
                    "best_roi": round(best_roi, 4),
                    "note": "Budget may be insufficient for some bid levels",
                }
        else:
            recommendation = {}

        return {
            "bid_levels": [round(base_bid * m, 2) for m in bid_multipliers],
            "results": results,
            "optimal_bid": round(optimal_bid, 2),
            "recommendation": recommendation,
        }
