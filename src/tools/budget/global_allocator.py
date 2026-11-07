"""GlobalBudgetAllocator - cross-channel budget allocation.

Maximise total expected incremental GMV subject to the total budget and
per-channel caps, over the discrete (spend, gmv) grids produced by
``ChannelResponseModel``. Primary solver: PuLP + HiGHS MIP (single-
selection formulation). Fallback: exhaustive enumeration — exact at this
scale (4 channels x 11 levels = 12^4 combinations) and dependency-free,
so the allocator degrades in *method*, never in *correctness*.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pulp

    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    logger.warning("PuLP not available; global allocation falls back to enumeration")


class GlobalBudgetAllocator:
    """Allocate one budget across channel grids (concave curves)."""

    def allocate(
        self,
        channel_grids: dict[str, list[tuple[float, float]]],
        total_budget: float,
    ) -> dict[str, Any]:
        if total_budget <= 0:
            return {"error": "total_budget must be positive"}
        if not channel_grids:
            return {"error": "channel_grids is empty"}

        if HAS_PULP:
            try:
                result = self._solve_ip(channel_grids, total_budget)
                result["solver_method"] = "ip-highs"
                return result
            except Exception as exc:  # solver trouble must never break the pipeline
                logger.warning("[global_allocator] IP failed (%s); enumerating", exc)

        result = self._enumerate(channel_grids, total_budget)
        result["solver_method"] = "enumeration"
        return result

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _solve_ip(
        self, channel_grids: dict[str, list[tuple[float, float]]], total_budget: float
    ) -> dict[str, Any]:
        """Single-selection MIP: each channel picks at most one grid level."""
        prob = pulp.LpProblem("global_budget", pulp.LpMaximize)
        x: dict[tuple[str, int], Any] = {}
        for channel, grid in channel_grids.items():
            for level in range(len(grid)):
                x[(channel, level)] = pulp.LpVariable(f"x_{channel}_{level}", cat="Binary")

        prob += pulp.lpSum(
            grid[level][1] * x[(channel, level)]
            for channel, grid in channel_grids.items()
            for level in range(len(grid))
        )
        # at most one level per channel
        for channel, grid in channel_grids.items():
            prob += pulp.lpSum(x[(channel, l)] for l in range(len(grid))) <= 1
        # total spend within budget
        prob += (
            pulp.lpSum(
                grid[level][0] * x[(channel, level)]
                for channel, grid in channel_grids.items()
                for level in range(len(grid))
            )
            <= total_budget
        )

        prob.solve(pulp.HiGHS(msg=False))  # type: ignore[attr-defined]
        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError(f"solver status {pulp.LpStatus[prob.status]}")

        allocation: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}
        for channel, grid in channel_grids.items():
            # HiGHS returns binaries as 0.99999999999…, never exactly 1 —
            # threshold, don't equality-test.
            chosen = next(
                (l for l in range(len(grid)) if (x[(channel, l)].value() or 0) > 0.5), 0
            )
            spend, gmv = grid[chosen]
            allocation[channel] = spend
            details[channel] = {
                "spend": spend,
                "expected_incremental_gmv": gmv,
                "expected_roi": round(gmv / spend, 4) if spend > 0 else 0.0,
                "budget_share": round(spend / total_budget, 4),
            }
        return self._assemble(allocation, details, total_budget)

    def _enumerate(
        self, channel_grids: dict[str, list[tuple[float, float]]], total_budget: float
    ) -> dict[str, Any]:
        """Exact fallback: iterate every combination (deterministic order)."""
        channels = list(channel_grids)
        best: tuple[float, dict[str, int]] = (0.0, {c: 0 for c in channels})
        for combo in itertools.product(*(range(len(channel_grids[c])) for c in channels)):
            spend = sum(channel_grids[c][l][0] for c, l in zip(channels, combo))
            if spend > total_budget + 1e-9:
                continue
            value = sum(channel_grids[c][l][1] for c, l in zip(channels, combo))
            if value > best[0] + 1e-9:  # strict: first-best wins ties
                best = (value, dict(zip(channels, combo)))

        allocation = {c: channel_grids[c][l][0] for c, l in best[1].items()}
        details = {
            c: {
                "spend": channel_grids[c][l][0],
                "expected_incremental_gmv": channel_grids[c][l][1],
                "expected_roi": (
                    round(channel_grids[c][l][1] / channel_grids[c][l][0], 4)
                    if channel_grids[c][l][0] > 0
                    else 0.0
                ),
                "budget_share": round(channel_grids[c][l][0] / total_budget, 4),
            }
            for c, l in best[1].items()
        }
        return self._assemble(allocation, details, total_budget)

    @staticmethod
    def _assemble(
        allocation: dict[str, float], details: dict[str, dict[str, Any]], total_budget: float
    ) -> dict[str, Any]:
        total_gmv = round(sum(d["expected_incremental_gmv"] for d in details.values()), 2)
        spent = round(sum(allocation.values()), 2)
        return {
            "status": "optimal",
            "allocation": allocation,
            "channel_details": details,
            "total_spend": spent,
            "unallocated": round(total_budget - spent, 2),
            "expected_incremental_gmv": total_gmv,
            "expected_roi": round(total_gmv / spent, 4) if spent > 0 else 0.0,
        }
