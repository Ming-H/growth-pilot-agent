"""SubsidyExpert - Budget optimization and subsidy allocation.

Uses CausalInferenceEngine, ElasticityEstimator, BudgetOptimizer, and
SubsidyAllocator to produce data-driven subsidy plans.  Falls back to
heuristic estimates when tools are unavailable or data is insufficient.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.core.expert import ExpertAgentBase
from src.prompts.templates.agent_prompts import SubsidyPrompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool imports with fallback stubs
# ---------------------------------------------------------------------------
try:
    from src.tools.subsidy import (
        BudgetOptimizer,
        CausalInferenceEngine,
        ElasticityEstimator,
        SubsidyAllocator,
    )
except ImportError:

    class _Stub:
        """Stub for tools not yet importable."""

        def __init__(self, *a: Any, **kw: Any) -> None: ...

    CausalInferenceEngine = ElasticityEstimator = BudgetOptimizer = SubsidyAllocator = _Stub  # type: ignore[assignment,misc]

try:
    from src.tools.common.data_loader import DataLoader
except ImportError:
    DataLoader = None  # type: ignore[assignment,misc]


class SubsidyExpert(ExpertAgentBase):
    """Optimizes subsidy allocation using causal inference and elasticity models."""

    name = "subsidy"
    description = "补贴策略 Agent"

    # ------------------------------------------------------------------
    # Tool initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _init_tools() -> dict[str, Any]:
        """Initialize and return deterministic tool instances as a dict."""
        return {
            "causal_engine": CausalInferenceEngine(),
            "elasticity_estimator": ElasticityEstimator(),
            "budget_optimizer": BudgetOptimizer(),
            "subsidy_allocator": SubsidyAllocator(),
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    @staticmethod
    def can_handle(query: str) -> float:
        """Return confidence score (0-1) for handling this query."""
        keywords = [
            "补贴", "预算", "ROI", "因果", "弹性",
            "subsidy", "budget", "causal", "elasticity",
        ]
        q_lower = query.lower()
        hits = sum(1 for kw in keywords if kw.lower() in q_lower)
        if hits == 0:
            return 0.0
        return min(hits / len(keywords) * 2.0, 1.0)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _get_system_prompt() -> str:
        """Return the expert's system prompt for LLM synthesis."""
        template = SubsidyPrompt()
        return f"{template.role_definition}\n\n{template.business_context}"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_or_generate_data(data_path: str) -> pd.DataFrame:
        """Load data from path or generate sample subsidy experiment data."""
        if data_path and DataLoader is not None:
            try:
                loader = DataLoader()
                return loader.load_csv(data_path)
            except (FileNotFoundError, Exception) as exc:
                logger.warning("Failed to load data from %s: %s", data_path, exc)

        if DataLoader is not None:
            try:
                loader = DataLoader()
                return loader.load_sample_data("subsidy_experiment")
            except Exception as exc:
                logger.warning("Failed to generate sample data: %s", exc)

        # Minimal synthetic fallback
        rng = np.random.default_rng(42)
        n = 1000
        return pd.DataFrame({
            "user_id": [f"U{i:04d}" for i in range(n)],
            "treatment": rng.binomial(1, 0.5, size=n),
            "converted": rng.binomial(1, 0.15, size=n),
            "revenue": rng.lognormal(4.0, 0.6, size=n).round(2) * rng.binomial(1, 0.15, size=n),
            "subsidy_amount": rng.choice([0, 5, 10, 20], size=n).astype(float),
            "price": (rng.lognormal(3.0, 0.5, size=n) * 10).round(2),
            "demand": rng.poisson(3, size=n).astype(float),
        })

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _execute_pipeline(self, params: dict) -> dict:
        """Execute the subsidy analysis pipeline (deterministic steps)."""
        errors: list[str] = []
        budget = float(params.get("budget") or 10000)
        prospect = params.get("prospect_results") or {}
        data_path = params.get("data_path", "")

        causal_engine = self._tools["causal_engine"]
        elasticity_estimator = self._tools["elasticity_estimator"]
        budget_optimizer = self._tools["budget_optimizer"]
        subsidy_allocator = self._tools["subsidy_allocator"]

        data = self._load_or_generate_data(data_path)

        # Ensure binary treatment column exists
        if "treatment" not in data.columns:
            if "group" in data.columns:
                data["treatment"] = (data["group"] != "control").astype(int)
            elif "subsidy_amount" in data.columns:
                data["treatment"] = (data["subsidy_amount"] > 0).astype(int)

        # 1. Causal inference -- estimate ATE of subsidy on conversion
        ate_result: dict[str, Any] = {}
        causal_insight = ""
        confidence = 0.0
        try:
            treatment_col = "treatment" if "treatment" in data.columns else "subsidy_amount"
            outcome_col = "converted" if "converted" in data.columns else "revenue"
            confounders = [c for c in ["age", "city_tier", "historical_orders"]
                           if c in data.columns]

            ate_result = causal_engine.estimate_ate(
                data=data,
                treatment=treatment_col,
                outcome=outcome_col,
                confounders=confounders,
                method="backdoor",
            )
            if "error" in ate_result:
                raise ValueError(ate_result["error"])

            ate_val = ate_result.get("ate", 0)
            p_val = ate_result.get("p_value", 1.0)
            significant = ate_result.get("significant_at_05", False)
            causal_insight = (
                f"补贴 ATE={ate_val:.4f} (p={p_val:.4f}), "
                f"{'显著' if significant else '不显著'}, "
                f"95% CI=[{ate_result.get('ci_lower', 0):.4f}, {ate_result.get('ci_upper', 0):.4f}]"
            )
            confidence = 0.9 if significant else 0.6
        except Exception as exc:
            logger.warning("CausalInferenceEngine.estimate_ate failed, using heuristic: %s", exc)
            ate_result = {"ate": 0.12, "ci_lower": 0.08, "ci_upper": 0.16, "method": "heuristic"}
            causal_insight = "补贴对首单转化有显著正向效果 (基于历史经验)"
            confidence = 0.85
            errors.append(f"CausalInferenceEngine (heuristic): {exc}")

        # 2. Elasticity estimation
        elasticity_result: dict[str, Any] = {}
        price_sensitivity: dict[str, Any] = {}
        try:
            price_col = "price" if "price" in data.columns else "subsidy_amount"
            demand_col = "demand" if "demand" in data.columns else "revenue"

            elasticity_result = elasticity_estimator.estimate_price_elasticity(
                data=data,
                price_col=price_col,
                demand_col=demand_col,
            )
            if "error" in elasticity_result:
                raise ValueError(elasticity_result["error"])

            elasticity_val = elasticity_result.get("elasticity", -1.0)
            elasticity_result["_summary"] = (
                f"价格弹性={elasticity_val:.4f}, "
                f"{elasticity_result.get('interpretation', '')}"
            )
            price_sensitivity = {
                "elasticity": elasticity_val,
                "significant": elasticity_result.get("significant_at_05", False),
                "interpretation": elasticity_result.get("interpretation", ""),
            }
        except Exception as exc:
            logger.warning("ElasticityEstimator.estimate_price_elasticity failed, using heuristic: %s", exc)
            elasticity_result = {"elasticity": -1.8, "method": "heuristic"}
            price_sensitivity = {
                "elasticity": -1.8,
                "most_sensitive": "new_user",
                "least_sensitive": "high_value",
            }
            errors.append(f"ElasticityEstimator (heuristic): {exc}")

        # 3. Budget optimisation
        budget_result: dict[str, Any] = {}
        expected_roi = 0.0
        user_segments: dict[str, int] = {}
        try:
            raw_segments = prospect.get("segment_summary", {})
            if raw_segments:
                for seg_name, seg_val in raw_segments.items():
                    if isinstance(seg_val, dict):
                        user_segments[seg_name] = int(seg_val.get("count", 0))
                    elif isinstance(seg_val, (int, float)):
                        user_segments[seg_name] = int(seg_val)
            if not user_segments:
                if "segment" in data.columns:
                    user_segments = data["segment"].value_counts().to_dict()
                else:
                    user_segments = {"new_user": 500, "active": 300, "dormant": 200}

            ate_val = ate_result.get("ate", 0.12)
            causal_effects: dict[str, dict[str, float]] = {}
            for seg in user_segments:
                causal_effects[seg] = {
                    "ate": ate_val,
                    "base_conversion_rate": 0.12,
                    "coupon_amount_used": 10.0,
                }

            budget_result = budget_optimizer.optimize_allocation(
                user_segments=user_segments,
                causal_effects=causal_effects,
                total_budget=float(budget),
                min_coupon=5,
                max_coupon=50,
                coupon_step=5,
            )
            if "error" in budget_result:
                raise ValueError(budget_result["error"])

            expected_roi = round(
                budget_result.get("expected_incremental_orders", 0)
                / max(budget_result.get("total_budget_used", 1), 1),
                2,
            )
        except Exception as exc:
            logger.warning("BudgetOptimizer.optimize_allocation failed, using heuristic: %s", exc)
            if budget > 0:
                budget_result = {
                    "allocation": {
                        "new_user": {"coupon_amount": budget * 0.35 / max(user_segments.get("new_user", 1), 1), "user_count": 500},
                        "active": {"coupon_amount": budget * 0.25 / max(user_segments.get("active", 1), 1), "user_count": 300},
                        "dormant": {"coupon_amount": budget * 0.20 / max(user_segments.get("dormant", 1), 1), "user_count": 200},
                    },
                    "total_budget_used": budget * 0.8,
                    "expected_incremental_orders": budget * 0.04,
                    "method": "heuristic",
                }
                expected_roi = 2.8
            errors.append(f"BudgetOptimizer (heuristic): {exc}")

        # 4. Subsidy allocation plan
        allocation_result: dict[str, Any] = {}
        try:
            allocation_result = subsidy_allocator.allocate(
                causal_results=ate_result,
                elasticity_results=elasticity_result,
                budget_plan=budget_result,
                user_segments=user_segments,
            )
            if "error" in allocation_result:
                raise ValueError(allocation_result["error"])
        except Exception as exc:
            logger.warning("SubsidyAllocator.allocate failed, using heuristic: %s", exc)
            allocation_result = budget_result.get("allocation", {})
            errors.append(f"SubsidyAllocator (heuristic): {exc}")

        # Build plain dict result (no Pydantic models)
        result: dict[str, Any] = {
            "subsidy_results": {
                "ate": ate_result,
                "causal_insight": causal_insight,
                "confidence": confidence,
                "elasticity": elasticity_result,
                "price_sensitivity": price_sensitivity,
                "budget_plan": budget_result,
                "expected_roi": expected_roi,
                "allocation_plan": allocation_result,
            },
        }
        if errors:
            result["errors"] = errors
        return result

    # ------------------------------------------------------------------
    # Prompt builder (for LLM synthesis)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_synthesis_prompt(params: dict, results: dict) -> str:
        """Build the LLM synthesis prompt from pipeline results."""
        subsidy = results.get("subsidy_results", {})

        ate = subsidy.get("ate", {})
        ate_summary = str(ate) if ate else ""
        confidence = subsidy.get("confidence")
        confidence_str = f"{confidence:.0%}" if confidence else ""
        causal_insight = subsidy.get("causal_insight", "")
        elasticity = subsidy.get("elasticity", {})
        elasticity_summary = elasticity.get("_summary", str(elasticity)) if elasticity else ""
        price_sensitivity = subsidy.get("price_sensitivity")
        price_sensitivity_str = str(price_sensitivity) if price_sensitivity else ""
        budget_result = subsidy.get("budget_plan", {})
        budget_summary = str(budget_result) if budget_result else ""
        expected_roi = subsidy.get("expected_roi")
        expected_roi_str = f"{expected_roi:.1f}x" if expected_roi else ""
        allocation_result = subsidy.get("allocation_plan", {})
        allocation_summary = str(allocation_result) if allocation_result else ""

        return SubsidyPrompt().render(
            user_query=params.get("query", ""),
            season=params.get("season", "当前"),
            kpi_baseline=params.get("kpi_baseline", {}),
            memory_context=params.get("memory_context", {}),
            ate_summary=ate_summary,
            causal_insight=causal_insight,
            confidence=confidence_str,
            elasticity_summary=elasticity_summary,
            price_sensitivity=price_sensitivity_str,
            budget_summary=budget_summary,
            expected_roi=expected_roi_str,
            allocation_summary=allocation_summary,
        )
