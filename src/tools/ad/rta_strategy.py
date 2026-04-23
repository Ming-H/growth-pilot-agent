"""RTA (Real-Time API) bidding strategy for external ad platforms.

Makes real-time bid/no-bid decisions for ad impressions based on prospect
scores, estimated conversion rates, and target CPA constraints.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from src.tools.registry import ToolRegistry


@ToolRegistry.register("rta_strategy")
class RTAStrategy:
    """Real-Time API bidding strategy engine.

    Evaluates incoming ad impression opportunities and decides whether to bid,
    and at what price, based on user features, prospect scoring, and campaign
    performance targets.
    """

    def should_bid(
        self,
        user_features: dict[str, Any],
        prospect_scores: dict[str, float],
        bid_floor: float,
        target_cpa: float,
    ) -> tuple[bool, float]:
        """Decide whether to bid on an impression and at what price.

        Args:
            user_features: Dict with keys like 'age', 'city_tier', 'historical_orders',
                           'days_since_last_order', 'device_type', etc.
            prospect_scores: Dict with scores like 'conversion_prob', 'churn_risk',
                             'ltv_estimate', 'intent_score'.
            bid_floor: Minimum bid price required by the ad platform (CNY).
            target_cpa: Target cost-per-acquisition for the campaign (CNY).

        Returns:
            Tuple of (should_bid: bool, bid_price: float).
            bid_price is 0.0 when should_bid is False.
        """
        if bid_floor <= 0:
            return False, 0.0
        if target_cpa <= 0:
            return False, 0.0

        cvr = self._estimate_cvr(user_features, prospect_scores)
        user_value = self._estimate_user_value(user_features, prospect_scores)

        # Bid = CVR * user_value (expected value per impression)
        bid_price = cvr * user_value

        # Apply a conservative margin to avoid overbidding
        margin = 0.85
        adjusted_bid = bid_price * margin

        # Check profitability: expected cost per acquisition must be below target
        if cvr > 0:
            expected_cpa = adjusted_bid / cvr
            if expected_cpa > target_cpa:
                return False, 0.0
        else:
            return False, 0.0

        # Check against bid floor
        if adjusted_bid < bid_floor:
            # Only bid if the gap is small enough (within 15% of floor)
            if adjusted_bid >= bid_floor * 0.85:
                return True, bid_floor
            return False, 0.0

        return True, round(adjusted_bid, 4)

    def build_rta_decision_rules(self, historical_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a decision tree of bidding rules from historical win/loss data.

        Analyzes historical bidding outcomes to create segment-based rules that
        can be deployed to RTA endpoints for fast decisions.

        Args:
            historical_data: List of dicts, each containing:
                - 'user_features': dict of user attributes
                - 'outcome': 'win' | 'loss' | 'no_bid'
                - 'cpa': actual CPA if converted, else None
                - 'revenue': revenue from conversion, else 0

        Returns:
            Dict with keys:
                - 'rules': list of segment rules
                - 'segments': segment statistics
                - 'overall_metrics': aggregate performance
        """
        if not historical_data:
            return {"rules": [], "segments": {}, "overall_metrics": {}}

        total = len(historical_data)
        wins = [d for d in historical_data if d.get("outcome") == "win"]
        losses = [d for d in historical_data if d.get("outcome") == "loss"]

        # Segment by city_tier
        tier_groups: dict[str, list[dict]] = {}
        for d in historical_data:
            features = d.get("user_features", {})
            tier = str(features.get("city_tier", "unknown"))
            tier_groups.setdefault(tier, []).append(d)

        segments: dict[str, dict[str, Any]] = {}
        for tier, group in tier_groups.items():
            group_wins = [d for d in group if d.get("outcome") == "win"]
            win_rate = len(group_wins) / len(group) if group else 0
            avg_cpa = np.mean([d["cpa"] for d in group_wins if d.get("cpa")]) if any(d.get("cpa") for d in group_wins) else 0.0
            avg_revenue = np.mean([d.get("revenue", 0) for d in group_wins]) if group_wins else 0.0
            segments[tier] = {
                "count": len(group),
                "win_rate": round(win_rate, 4),
                "avg_cpa": round(float(avg_cpa), 2),
                "avg_revenue": round(float(avg_revenue), 2),
                "roi": round(float(avg_revenue / avg_cpa), 4) if avg_cpa > 0 else 0.0,
            }

        # Build decision rules based on segments
        rules: list[dict[str, Any]] = []
        for tier, stats in sorted(segments.items()):
            if stats["roi"] > 1.0 and stats["win_rate"] > 0.05:
                action = "bid_aggressive"
                bid_modifier = min(1.3, stats["roi"] * 0.5)
            elif stats["roi"] > 0.5 and stats["win_rate"] > 0.03:
                action = "bid_conservative"
                bid_modifier = 0.8
            elif stats["win_rate"] > 0.02:
                action = "bid_min"
                bid_modifier = 0.5
            else:
                action = "skip"
                bid_modifier = 0.0

            rules.append({
                "segment": f"city_tier_{tier}",
                "action": action,
                "bid_modifier": round(bid_modifier, 2),
                "win_rate": stats["win_rate"],
                "roi": stats["roi"],
            })

        # Segment by intent_score ranges
        intent_groups: dict[str, list[dict]] = {"high": [], "medium": [], "low": []}
        for d in historical_data:
            features = d.get("user_features", {})
            intent = features.get("intent_score", 0)
            if intent >= 0.7:
                intent_groups["high"].append(d)
            elif intent >= 0.4:
                intent_groups["medium"].append(d)
            else:
                intent_groups["low"].append(d)

        for intent_level, group in intent_groups.items():
            if not group:
                continue
            group_wins = [d for d in group if d.get("outcome") == "win"]
            win_rate = len(group_wins) / len(group) if group else 0
            avg_cpa = np.mean([d["cpa"] for d in group_wins if d.get("cpa")]) if any(d.get("cpa") for d in group_wins) else 0.0
            avg_revenue = np.mean([d.get("revenue", 0) for d in group_wins]) if group_wins else 0.0
            roi = float(avg_revenue / avg_cpa) if avg_cpa > 0 else 0.0

            if roi > 0.8 and win_rate > 0.03:
                action = "bid_aggressive" if intent_level == "high" else "bid_conservative"
                bid_modifier = min(1.5, roi * 0.6) if intent_level == "high" else 0.9
            elif win_rate > 0.02:
                action = "bid_min"
                bid_modifier = 0.5
            else:
                action = "skip"
                bid_modifier = 0.0

            rules.append({
                "segment": f"intent_{intent_level}",
                "action": action,
                "bid_modifier": round(bid_modifier, 2),
                "win_rate": round(win_rate, 4),
                "roi": round(roi, 4),
            })

        overall_win_rate = len(wins) / total if total > 0 else 0
        overall_revenue = sum(d.get("revenue", 0) for d in wins)
        overall_cpa_list = [d["cpa"] for d in wins if d.get("cpa")]
        overall_cpa = float(np.mean(overall_cpa_list)) if overall_cpa_list else 0.0

        overall_metrics = {
            "total_impressions": total,
            "total_wins": len(wins),
            "total_losses": len(losses),
            "win_rate": round(overall_win_rate, 4),
            "overall_revenue": round(overall_revenue, 2),
            "overall_avg_cpa": round(overall_cpa, 2),
        }

        return {
            "rules": rules,
            "segments": segments,
            "overall_metrics": overall_metrics,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_cvr(self, user_features: dict[str, Any], prospect_scores: dict[str, float]) -> float:
        """Estimate conversion rate from user features and prospect scores."""
        base_cvr = prospect_scores.get("conversion_prob", 0.01)
        intent = prospect_scores.get("intent_score", 0.0)

        # Boost CVR for returning users with recent activity
        days_since = user_features.get("days_since_last_order", 999)
        if days_since <= 7:
            recency_boost = 1.5
        elif days_since <= 30:
            recency_boost = 1.2
        else:
            recency_boost = 0.9

        # Historical order count boost
        order_count = user_features.get("historical_orders", 0)
        if order_count >= 5:
            frequency_boost = 1.3
        elif order_count >= 2:
            frequency_boost = 1.1
        else:
            frequency_boost = 1.0

        estimated_cvr = base_cvr * recency_boost * frequency_boost
        # Blend with intent score
        estimated_cvr = 0.6 * estimated_cvr + 0.4 * (intent * 0.05)

        return max(0.0, min(estimated_cvr, 0.5))

    def _estimate_user_value(self, user_features: dict[str, Any], prospect_scores: dict[str, float]) -> float:
        """Estimate the expected revenue value of a user if they convert."""
        ltv = prospect_scores.get("ltv_estimate", 50.0)

        # Adjust by city tier (higher tier cities tend to have higher order values)
        tier = user_features.get("city_tier", 2)
        tier_multiplier = {1: 1.4, 2: 1.0, 3: 0.8, 4: 0.6}.get(int(tier), 1.0)

        # Adjust by device type
        device = user_features.get("device_type", "android")
        device_multiplier = {"ios": 1.15, "android": 1.0, "web": 0.9}.get(device, 1.0)

        return ltv * tier_multiplier * device_multiplier
