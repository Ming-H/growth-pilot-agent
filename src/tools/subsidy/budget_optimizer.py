"""BudgetOptimizer - optimize subsidy budget allocation using PuLP.

Uses integer programming to maximise expected incremental orders subject
to budget, per-segment min/max constraints, and discrete coupon amounts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    logger.warning("PuLP not available; budget optimisation will use greedy fallback")


@ToolRegistry.register("budget_optimizer")
class BudgetOptimizer:
    """Optimise subsidy budget allocation across user segments."""

    def optimize_allocation(
        self,
        user_segments: dict[str, int],
        causal_effects: dict[str, dict[str, float]],
        total_budget: float,
        min_coupon: float = 5,
        max_coupon: float = 50,
        coupon_step: float = 5,
    ) -> dict[str, Any]:
        """Allocate budget to maximise expected incremental orders.

        Args:
            user_segments: mapping of segment_name -> user_count.
            causal_effects: mapping of segment_name -> {
                "ate": float,  # average treatment effect (incremental order rate per yuan)
                "base_conversion_rate": float,
                ...
            }
            total_budget: total subsidy budget in yuan.
            min_coupon: minimum coupon amount per user (yuan).
            max_coupon: maximum coupon amount per user (yuan).
            coupon_step: granularity of coupon amounts (yuan).

        Returns:
            Optimal allocation plan with per-segment coupon details.
        """
        if not user_segments:
            return {"error": "user_segments is empty"}

        if total_budget <= 0:
            return {"error": "total_budget must be positive"}

        segments = list(user_segments.keys())
        possible_amounts = np.arange(min_coupon, max_coupon + coupon_step / 2, coupon_step)

        if len(possible_amounts) == 0:
            return {"error": "no valid coupon amounts in range"}

        if HAS_PULP:
            return self._optimize_pulp(
                segments, user_segments, causal_effects, total_budget, possible_amounts
            )
        else:
            return self._optimize_greedy(
                segments, user_segments, causal_effects, total_budget, possible_amounts
            )

    def sensitivity_analysis(
        self,
        optimal_plan: dict[str, Any],
        user_segments: dict[str, int],
        causal_effects: dict[str, dict[str, float]],
        budget_range: tuple[float, float] | list[float] | None = None,
        n_points: int = 10,
    ) -> dict[str, Any]:
        """Run sensitivity analysis across different budget levels.

        Args:
            optimal_plan: result from optimize_allocation().
            user_segments: segment -> user_count.
            causal_effects: segment -> causal effect dict.
            budget_range: (min_budget, max_budget) or list of budgets.
                Defaults to 50% to 200% of the original budget.
            n_points: number of points if budget_range is a tuple.

        Returns:
            Sensitivity analysis results.
        """
        original_budget = optimal_plan.get("total_budget_used", 0)
        if original_budget <= 0:
            original_budget = optimal_plan.get("total_budget", 0)

        if budget_range is None:
            budget_range = (original_budget * 0.5, original_budget * 2.0)

        if isinstance(budget_range, (list, tuple)) and len(budget_range) == 2:
            budgets = np.linspace(budget_range[0], budget_range[1], n_points).tolist()
        elif isinstance(budget_range, list):
            budgets = budget_range
        else:
            budgets = [original_budget]

        results: list[dict[str, Any]] = []
        for budget in budgets:
            plan = self.optimize_allocation(
                user_segments, causal_effects, float(budget)
            )
            if "error" not in plan:
                results.append(
                    {
                        "budget": round(float(budget), 2),
                        "expected_incremental_orders": plan.get(
                            "expected_incremental_orders", 0
                        ),
                        "total_cost": plan.get("total_budget_used", 0),
                        "segments_served": plan.get("segments_served", 0),
                        "efficiency": round(
                            plan.get("expected_incremental_orders", 0)
                            / max(plan.get("total_budget_used", 1), 1),
                            6,
                        ),
                    }
                )

        # Find budget with best efficiency
        best_efficiency = max(results, key=lambda x: x["efficiency"]) if results else None

        # Diminishing returns check
        if len(results) >= 3:
            efficiencies = [r["efficiency"] for r in results]
            has_diminishing = efficiencies[0] > efficiencies[-1]
        else:
            has_diminishing = None

        return {
            "original_budget": round(original_budget, 2),
            "sensitivity_results": results,
            "best_efficiency_budget": best_efficiency,
            "has_diminishing_returns": has_diminishing,
            "budget_step": round(
                float(budgets[1] - budgets[0]) if len(budgets) > 1 else 0, 2
            ),
        }

    # ------------------------------------------------------------------
    # Internal: PuLP-based optimisation
    # ------------------------------------------------------------------

    def _optimize_pulp(
        self,
        segments: list[str],
        user_segments: dict[str, int],
        causal_effects: dict[str, dict[str, float]],
        total_budget: float,
        possible_amounts: np.ndarray,
    ) -> dict[str, Any]:
        """Integer programming via PuLP."""
        prob = pulp.LpProblem("SubsidyBudgetOptimisation", pulp.LpMaximize)

        # Decision variables: x[s][a] = 1 if segment s gets coupon amount a
        x: dict[str, dict[float, pulp.LpVariable]] = {}
        for s in segments:
            x[s] = {}
            for a in possible_amounts:
                var_name = f"x_{s}_{int(a)}"
                x[s][a] = pulp.LpVariable(var_name, cat="Binary")

        # Objective: maximise expected incremental orders
        # Expected incremental for segment s with amount a =
        #   user_count * (ate * a / avg_coupon_in_causal) * (a / aov_proxy)
        # Simplified: user_count * ate * amount as a proxy
        obj_terms = []
        for s in segments:
            ate = causal_effects.get(s, {}).get("ate", 0.01)
            n_users = user_segments[s]
            for a in possible_amounts:
                # Expected incremental orders = n_users * ate * (amount / reference_amount)
                # reference_amount normalises the ATE to per-yuan effect
                ref = causal_effects.get(s, {}).get("coupon_amount_used", 10.0)
                incremental = n_users * ate * (a / ref)
                obj_terms.append(incremental * x[s][a])

        prob += pulp.lpSum(obj_terms), "ExpectedIncrementalOrders"

        # Constraint 1: each segment gets exactly one coupon amount
        for s in segments:
            prob += pulp.lpSum(x[s][a] for a in possible_amounts) <= 1, f"one_amount_{s}"

        # Constraint 2: total budget
        budget_terms = []
        for s in segments:
            n_users = user_segments[s]
            for a in possible_amounts:
                budget_terms.append(n_users * a * x[s][a])
        prob += pulp.lpSum(budget_terms) <= total_budget, "TotalBudget"

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract solution
        allocation: dict[str, Any] = {}
        total_cost = 0.0
        total_incremental = 0.0

        for s in segments:
            allocated_amount = 0.0
            for a in possible_amounts:
                if pulp.value(x[s][a]) is not None and pulp.value(x[s][a]) > 0.5:
                    allocated_amount = float(a)
                    break

            if allocated_amount > 0:
                n_users = user_segments[s]
                ate = causal_effects.get(s, {}).get("ate", 0.01)
                ref = causal_effects.get(s, {}).get("coupon_amount_used", 10.0)
                inc_orders = n_users * ate * (allocated_amount / ref)
                cost = n_users * allocated_amount

                allocation[s] = {
                    "coupon_amount": allocated_amount,
                    "user_count": n_users,
                    "total_cost": round(cost, 2),
                    "expected_incremental_orders": round(inc_orders, 2),
                    "roi": round(inc_orders / cost * 100, 2) if cost > 0 else 0.0,
                }
                total_cost += cost
                total_incremental += inc_orders

        return {
            "status": pulp.LpStatus[prob.status],
            "allocation": allocation,
            "total_budget": total_budget,
            "total_budget_used": round(total_cost, 2),
            "budget_utilization": round(total_cost / total_budget, 4) if total_budget > 0 else 0.0,
            "expected_incremental_orders": round(total_incremental, 2),
            "segments_served": len(allocation),
            "method": "integer_programming",
        }

    # ------------------------------------------------------------------
    # Internal: Greedy fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _optimize_greedy(
        segments: list[str],
        user_segments: dict[str, int],
        causal_effects: dict[str, dict[str, float]],
        total_budget: float,
        possible_amounts: np.ndarray,
    ) -> dict[str, Any]:
        """Greedy allocation when PuLP is unavailable."""
        # Compute ROI per yuan per segment
        scored: list[dict[str, Any]] = []
        for s in segments:
            ate = causal_effects.get(s, {}).get("ate", 0.01)
            ref = causal_effects.get(s, {}).get("coupon_amount_used", 10.0)
            n_users = user_segments[s]

            for a in possible_amounts:
                cost = n_users * a
                inc_orders = n_users * ate * (a / ref)
                roi = inc_orders / cost if cost > 0 else 0
                scored.append(
                    {
                        "segment": s,
                        "amount": float(a),
                        "cost": cost,
                        "incremental_orders": inc_orders,
                        "roi": roi,
                    }
                )

        scored.sort(key=lambda x: x["roi"], reverse=True)

        allocation: dict[str, Any] = {}
        remaining_budget = total_budget
        total_cost = 0.0
        total_incremental = 0.0

        for item in scored:
            s = item["segment"]
            if s in allocation:
                continue
            cost = item["cost"]
            if cost <= remaining_budget:
                allocation[s] = {
                    "coupon_amount": item["amount"],
                    "user_count": user_segments[s],
                    "total_cost": round(cost, 2),
                    "expected_incremental_orders": round(item["incremental_orders"], 2),
                    "roi": round(item["roi"] * 100, 2),
                }
                remaining_budget -= cost
                total_cost += cost
                total_incremental += item["incremental_orders"]

        return {
            "status": "optimal" if len(allocation) == len(segments) else "partial",
            "allocation": allocation,
            "total_budget": total_budget,
            "total_budget_used": round(total_cost, 2),
            "budget_utilization": round(total_cost / total_budget, 4) if total_budget > 0 else 0.0,
            "expected_incremental_orders": round(total_incremental, 2),
            "segments_served": len(allocation),
            "method": "greedy",
        }
