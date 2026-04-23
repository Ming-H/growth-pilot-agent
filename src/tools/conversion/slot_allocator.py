"""SlotAllocator - allocate limited in-app slots (金刚位, Banner).

Uses a weighted scoring approach based on segment value, slot performance,
and capacity constraints to produce an optimal allocation plan.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default slot definitions for Didi freight app
DEFAULT_SLOTS = {
    "金刚位": {
        "total_capacity": 4,          # max items shown at once
        "impressions_per_day": 500000,
        "avg_ctr": 0.08,
    },
    "Banner_首页轮播": {
        "total_capacity": 6,
        "impressions_per_day": 300000,
        "avg_ctr": 0.04,
    },
    "Banner_发单页": {
        "total_capacity": 2,
        "impressions_per_day": 200000,
        "avg_ctr": 0.06,
    },
    "Banner_完成页": {
        "total_capacity": 2,
        "impressions_per_day": 100000,
        "avg_ctr": 0.05,
    },
}


@ToolRegistry.register("slot_allocator")
class SlotAllocator:
    """Allocate limited in-app slots across user segments."""

    def allocate_slots(
        self,
        user_segments: dict[str, dict[str, Any]],
        slot_capacity: dict[str, int] | None = None,
        performance_data: pd.DataFrame | dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        """Produce an allocation plan mapping segments to slots.

        Args:
            user_segments: mapping of segment_name -> {count, ltv, priority},
                where priority is 1-5 (5 = highest).
            slot_capacity: mapping of slot_name -> max items capacity.
                Defaults to DEFAULT_SLOTS capacities.
            performance_data: either a DataFrame with columns
                [slot, segment, ctr, conversion_rate] or a nested dict
                slot -> segment -> {ctr, conversion_rate}.

        Returns:
            Allocation plan dict with per-slot assignments and rationale.
        """
        if not user_segments:
            return {"error": "user_segments is empty", "allocation": {}}

        # Resolve capacities
        capacities = slot_capacity or {
            name: meta["total_capacity"] for name, meta in DEFAULT_SLOTS.items()
        }

        # Build performance lookup
        perf_lookup = self._build_performance_lookup(performance_data)

        # Score each (segment, slot) pair
        scores = self._score_all_pairs(user_segments, capacities, perf_lookup)

        # Allocate using greedy approach with capacity constraints
        allocation = self._greedy_allocate(scores, capacities)

        # Build result
        plan: dict[str, Any] = {}
        for slot_name, assignments in allocation.items():
            plan[slot_name] = {
                "capacity": capacities.get(slot_name, 1),
                "assignments": assignments,
                "utilization": f"{len(assignments)}/{capacities.get(slot_name, 1)}",
            }

        # Compute total expected impressions
        total_expected_impressions = 0
        total_expected_clicks = 0
        for slot_name, assignments in allocation.items():
            slot_meta = DEFAULT_SLOTS.get(slot_name, {})
            daily_imp = slot_meta.get("impressions_per_day", 100000)
            for a in assignments:
                imp_share = daily_imp / max(capacities.get(slot_name, 1), 1)
                ctr = a.get("expected_ctr", 0.05)
                total_expected_impressions += int(imp_share)
                total_expected_clicks += int(imp_share * ctr)

        return {
            "allocation": plan,
            "total_slots_used": sum(len(a) for a in allocation.values()),
            "total_slots_available": sum(capacities.values()),
            "expected_daily_impressions": total_expected_impressions,
            "expected_daily_clicks": total_expected_clicks,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_performance_lookup(
        self, performance_data: Any
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Build a lookup: slot -> segment -> {ctr, conversion_rate}."""
        lookup: dict[str, dict[str, dict[str, float]]] = {}

        if performance_data is None:
            # Use default estimates
            for slot_name, meta in DEFAULT_SLOTS.items():
                lookup[slot_name] = {
                    "__default__": {
                        "ctr": meta["avg_ctr"],
                        "conversion_rate": meta["avg_ctr"] * 0.3,
                    }
                }
            return lookup

        if isinstance(performance_data, pd.DataFrame):
            if performance_data.empty:
                return lookup
            for _, row in performance_data.iterrows():
                slot = str(row.get("slot", ""))
                segment = str(row.get("segment", "__default__"))
                lookup.setdefault(slot, {})[segment] = {
                    "ctr": float(row.get("ctr", 0.05)),
                    "conversion_rate": float(row.get("conversion_rate", 0.015)),
                }
            return lookup

        if isinstance(performance_data, dict):
            for slot, seg_data in performance_data.items():
                lookup[slot] = {}
                if isinstance(seg_data, dict):
                    for seg, metrics in seg_data.items():
                        if isinstance(metrics, dict):
                            lookup[slot][seg] = {
                                "ctr": float(metrics.get("ctr", 0.05)),
                                "conversion_rate": float(
                                    metrics.get("conversion_rate", 0.015)
                                ),
                            }
            return lookup

        return lookup

    def _score_all_pairs(
        self,
        user_segments: dict[str, dict[str, Any]],
        capacities: dict[str, int],
        perf_lookup: dict[str, dict[str, dict[str, float]]],
    ) -> list[dict[str, Any]]:
        """Score each (segment, slot) pair and return sorted list."""
        pairs: list[dict[str, Any]] = []

        for seg_name, seg_info in user_segments.items():
            count = seg_info.get("count", 0)
            ltv = seg_info.get("ltv", 50.0)
            priority = seg_info.get("priority", 3)

            for slot_name in capacities:
                # Get performance metrics
                slot_perf = perf_lookup.get(slot_name, {})
                metrics = slot_perf.get(
                    seg_name, slot_perf.get("__default__", {"ctr": 0.05, "conversion_rate": 0.015})
                )
                ctr = metrics["ctr"]
                cvr = metrics["conversion_rate"]

                # Composite score:
                #   user_volume * CTR * CVR * LTV * priority_weight
                score = count * ctr * cvr * ltv * (priority / 5.0)

                pairs.append(
                    {
                        "segment": seg_name,
                        "slot": slot_name,
                        "score": score,
                        "expected_ctr": ctr,
                        "expected_cvr": cvr,
                        "user_count": count,
                    }
                )

        pairs.sort(key=lambda x: x["score"], reverse=True)
        return pairs

    def _greedy_allocate(
        self,
        scored_pairs: list[dict[str, Any]],
        capacities: dict[str, int],
    ) -> dict[str, list[dict[str, Any]]]:
        """Greedy allocation: assign highest-scoring pairs first, respecting capacity."""
        allocation: dict[str, list[dict[str, Any]]] = {slot: [] for slot in capacities}
        remaining = dict(capacities)
        assigned_segments: set[str] = set()

        for pair in scored_pairs:
            slot = pair["slot"]
            segment = pair["segment"]

            if remaining.get(slot, 0) <= 0:
                continue
            if segment in assigned_segments:
                # Allow a segment to appear in multiple slots, but only once per slot
                if any(a["segment"] == segment for a in allocation[slot]):
                    continue

            allocation[slot].append(
                {
                    "segment": segment,
                    "score": round(pair["score"], 4),
                    "expected_ctr": pair["expected_ctr"],
                    "expected_cvr": pair["expected_cvr"],
                    "user_count": pair["user_count"],
                }
            )
            remaining[slot] -= 1
            assigned_segments.add(segment)

        return allocation
