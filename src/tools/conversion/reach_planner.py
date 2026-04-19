"""ReachPlanner - plan in-app reach strategy for Didi freight.

Decides optimal channel (金刚位/Banner/Push/SMS), timing, frequency cap,
and creative message for different user segments.
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Channel definitions with typical characteristics
CHANNELS = {
    "金刚位": {
        "visibility": 0.95,
        "intrusiveness": 0.3,
        "cost_per_exposure": 0.0,
        "best_for": ["active", "new_user", "high_value"],
        "max_daily_frequency": 1,
    },
    "Banner": {
        "visibility": 0.6,
        "intrusiveness": 0.2,
        "cost_per_exposure": 0.0,
        "best_for": ["active", "moderate", "new_user"],
        "max_daily_frequency": 3,
    },
    "Push": {
        "visibility": 0.4,
        "intrusiveness": 0.7,
        "cost_per_exposure": 0.01,
        "best_for": ["dormant", "at_risk", "moderate"],
        "max_daily_frequency": 2,
    },
    "SMS": {
        "visibility": 0.9,
        "intrusiveness": 0.9,
        "cost_per_exposure": 0.05,
        "best_for": ["dormant", "churned", "win_back"],
        "max_daily_frequency": 1,
    },
}

# Creative message templates per segment
CREATIVE_TEMPLATES: dict[str, list[str]] = {
    "new_user": [
        "首单立减，货运动动手指就搞定！",
        "新人专享优惠券已到账，限时使用！",
        "免费体验搬家/货运，新用户首单0元起！",
    ],
    "active": [
        "感谢使用滴滴货运，专属优惠为您准备！",
        "本周热门活动，老用户专属折扣！",
        "您有一张未使用的优惠券即将过期！",
    ],
    "moderate": [
        "好久不见，滴滴货运为您准备了专属回归优惠！",
        "限时福利：回老用户专享折扣！",
    ],
    "dormant": [
        "我们想您了！专属回归礼包等您领取！",
        "您的账户有一张大额优惠券待领取！",
    ],
    "at_risk": [
        "您有一张专属保留优惠券，快来看看！",
        "滴滴货运升级啦，新体验新优惠！",
    ],
    "high_value": [
        "尊享会员专属服务，更优价格更优体验！",
        "您的VIP专属折扣已到账！",
    ],
}


class ReachPlanner:
    """Plan in-app reach strategy for user segments."""

    def plan_reach_strategy(
        self,
        user_segments: dict[str, int],
        channel_performance: dict[str, dict[str, float]] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Decide channel, timing, frequency cap, and creative message per segment.

        Args:
            user_segments: mapping of segment_name -> user_count, e.g.
                {"new_user": 5000, "dormant": 2000, ...}
            channel_performance: optional override of default channel metrics, e.g.
                {"Push": {"ctr": 0.08, "conversion_rate": 0.02}, ...}
            constraints: optional constraints, e.g.
                {"max_daily_push": 10000, "max_daily_sms": 5000,
                 "budget": 500.0, "blacklist_channels": ["SMS"]}

        Returns:
            Strategy dict with per-segment plan including channel, timing,
            frequency, and creative.
        """
        if not user_segments:
            return {"error": "user_segments is empty", "strategies": {}}

        constraints = constraints or {}
        blacklist = set(constraints.get("blacklist_channels", []))
        budget = constraints.get("budget", float("inf"))
        max_daily_push = constraints.get("max_daily_push", float("inf"))
        max_daily_sms = constraints.get("max_daily_sms", float("inf"))

        # Merge default channel data with user-provided performance
        channel_scores = self._compute_channel_scores(channel_performance)

        strategies: dict[str, dict[str, Any]] = {}
        total_cost = 0.0

        for segment, count in user_segments.items():
            if count <= 0:
                continue

            # Rank channels for this segment
            ranked = self._rank_channels_for_segment(
                segment, count, channel_scores, blacklist
            )

            if not ranked:
                strategies[segment] = {
                    "channel": None,
                    "reason": "no suitable channel within constraints",
                }
                continue

            primary_channel = ranked[0]["channel"]
            secondary_channel = ranked[1]["channel"] if len(ranked) > 1 else None

            freq_cap = min(
                CHANNELS[primary_channel]["max_daily_frequency"],
                self._compute_frequency_cap(segment),
            )

            timing = self._compute_best_timing(segment)
            creative = self._select_creative(segment)

            exposure_cost = (
                CHANNELS[primary_channel]["cost_per_exposure"] * count * freq_cap
            )

            # Budget check
            if total_cost + exposure_cost > budget:
                # Try cheaper channels
                for candidate in ranked:
                    alt_cost = (
                        CHANNELS[candidate["channel"]]["cost_per_exposure"]
                        * count
                        * freq_cap
                    )
                    if total_cost + alt_cost <= budget:
                        primary_channel = candidate["channel"]
                        exposure_cost = alt_cost
                        break
                else:
                    strategies[segment] = {
                        "channel": None,
                        "reason": "budget exhausted",
                    }
                    continue

            # Volume caps
            if primary_channel == "Push" and count > max_daily_push:
                primary_channel = secondary_channel or "Banner"
            if primary_channel == "SMS" and count > max_daily_sms:
                primary_channel = secondary_channel or "Push"

            total_cost += exposure_cost

            strategies[segment] = {
                "channel": primary_channel,
                "secondary_channel": secondary_channel,
                "timing": timing,
                "frequency_cap": freq_cap,
                "creative_message": creative,
                "estimated_exposure": count,
                "estimated_cost": round(exposure_cost, 2),
                "channel_score": round(ranked[0]["score"], 4),
            }

        return {
            "strategies": strategies,
            "total_estimated_cost": round(total_cost, 2),
            "segments_planned": len(strategies),
            "total_users_targeted": sum(
                s.get("estimated_exposure", 0) for s in strategies.values()
            ),
        }

    def optimize_timing(self, user_activity: pd.DataFrame) -> dict[str, Any]:
        """Compute best time slots per user segment.

        Args:
            user_activity: DataFrame with columns [segment, hour (0-23), activity_count]

        Returns:
            Dict with best time slots per segment.
        """
        required_cols = {"segment", "hour", "activity_count"}
        if user_activity.empty or not required_cols.issubset(user_activity.columns):
            return {"error": "DataFrame must have columns: segment, hour, activity_count", "timing": {}}

        result: dict[str, Any] = {}
        for segment, group in user_activity.groupby("segment"):
            sorted_df = group.sort_values("activity_count", ascending=False)
            top_hours = sorted_df.head(3)["hour"].tolist()
            peak_hour = int(top_hours[0])

            result[segment] = {
                "peak_hour": peak_hour,
                "top_3_hours": [int(h) for h in top_hours],
                "recommended_push_window": self._format_time_window(peak_hour),
                "activity_distribution": (
                    group.set_index("hour")["activity_count"].to_dict()
                ),
            }

        return {"timing": result}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_channel_scores(
        self, channel_performance: dict[str, dict[str, float]] | None
    ) -> dict[str, float]:
        """Compute a composite score for each channel (higher = better)."""
        scores: dict[str, float] = {}
        for ch, meta in CHANNELS.items():
            base = meta["visibility"] * 0.4 + (1 - meta["intrusiveness"]) * 0.3
            if channel_performance and ch in channel_performance:
                perf = channel_performance[ch]
                ctr = perf.get("ctr", 0.05)
                cvr = perf.get("conversion_rate", 0.01)
                base += ctr * 2.0 + cvr * 5.0
            scores[ch] = base
        return scores

    def _rank_channels_for_segment(
        self,
        segment: str,
        count: int,
        channel_scores: dict[str, float],
        blacklist: set[str],
    ) -> list[dict[str, Any]]:
        """Return channels ranked by suitability for a segment."""
        ranked: list[dict[str, Any]] = []
        for ch, score in channel_scores.items():
            if ch in blacklist:
                continue
            meta = CHANNELS[ch]
            # Bonus if segment is in channel's best_for list
            segment_bonus = 1.5 if segment in meta["best_for"] else 0.5
            ranked.append(
                {"channel": ch, "score": score * segment_bonus, "base_score": score}
            )
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def _compute_frequency_cap(self, segment: str) -> int:
        """Heuristic frequency cap per segment."""
        caps: dict[str, int] = {
            "new_user": 3,
            "active": 2,
            "moderate": 2,
            "dormant": 1,
            "at_risk": 1,
            "high_value": 2,
            "churned": 1,
        }
        return caps.get(segment, 2)

    def _compute_best_timing(self, segment: str) -> dict[str, Any]:
        """Return recommended timing windows per segment (heuristic defaults)."""
        defaults: dict[str, dict[str, Any]] = {
            "new_user": {"hour": 10, "window": "10:00-12:00", "reason": "morning browsing peak"},
            "active": {"hour": 18, "window": "18:00-20:00", "reason": "evening commute planning"},
            "moderate": {"hour": 12, "window": "12:00-13:00", "reason": "lunch break"},
            "dormant": {"hour": 20, "window": "20:00-21:00", "reason": "evening leisure"},
            "at_risk": {"hour": 19, "window": "19:00-20:00", "reason": "evening engagement"},
            "high_value": {"hour": 9, "window": "09:00-10:00", "reason": "morning planning"},
            "churned": {"hour": 19, "window": "19:00-20:00", "reason": "evening leisure"},
        }
        return defaults.get(
            segment,
            {"hour": 12, "window": "12:00-13:00", "reason": "default midday"},
        )

    def _select_creative(self, segment: str) -> str:
        """Pick a creative message for the segment."""
        import random

        templates = CREATIVE_TEMPLATES.get(segment, CREATIVE_TEMPLATES.get("active", []))
        if not templates:
            return "滴滴货运优惠活动进行中！"
        return random.choice(templates)

    @staticmethod
    def _format_time_window(hour: int) -> str:
        """Format an hour into a readable time window string."""
        start = f"{hour:02d}:00"
        end_hour = min(hour + 2, 23)
        return f"{start}-{end_hour:02d}:00"
