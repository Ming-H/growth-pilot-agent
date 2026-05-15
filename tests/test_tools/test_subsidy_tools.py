"""Tests for subsidy tools: CausalInferenceEngine, ElasticityEstimator, BudgetOptimizer, SubsidyAllocator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# CausalInferenceEngine
# ---------------------------------------------------------------------------


class TestCausalInferenceEngine:
    """Tests for src.tools.subsidy.causal_engine.CausalInferenceEngine."""

    @pytest.fixture
    def engine(self):
        from src.tools.subsidy.causal_engine import CausalInferenceEngine
        return CausalInferenceEngine()

    @pytest.fixture
    def sample_causal_data(self):
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "treatment": np.random.binomial(1, 0.5, n),
            "outcome": np.random.normal(10, 3, n),
            "age": np.random.randint(18, 60, n),
            "income": np.random.normal(5000, 1000, n),
            "city_tier": np.random.choice([1, 2, 3], n),
        })

    def test_instantiate(self, engine):
        assert engine is not None

    def test_identify_causal_effect_returns_dict(self, engine, sample_causal_data):
        result = engine.identify_causal_effect(
            sample_causal_data,
            treatment="treatment",
            outcome="outcome",
            confounders=["age", "income"],
        )
        assert isinstance(result, dict)
        assert "estimand_type" in result
        assert result["estimand_type"] == "ATE"
        assert "treated_count" in result
        assert "control_count" in result

    def test_identify_causal_effect_empty_data(self, engine):
        result = engine.identify_causal_effect(
            pd.DataFrame(), treatment="t", outcome="y", confounders=[]
        )
        assert "error" in result

    def test_identify_causal_effect_missing_columns(self, engine):
        result = engine.identify_causal_effect(
            pd.DataFrame({"a": [1]}), treatment="missing", outcome="y", confounders=[]
        )
        assert "error" in result

    def test_estimate_ate_diff_in_means(self, engine, sample_causal_data):
        result = engine.estimate_ate(
            sample_causal_data, "treatment", "outcome",
            confounders=["age"], method="diff_in_means",
        )
        assert isinstance(result, dict)
        assert "ate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "method" in result

    def test_estimate_ate_backdoor(self, engine, sample_causal_data):
        result = engine.estimate_ate(
            sample_causal_data, "treatment", "outcome",
            confounders=["age", "income"], method="backdoor",
        )
        assert isinstance(result, dict)
        assert "ate" in result

    def test_estimate_ate_ipw(self, engine, sample_causal_data):
        result = engine.estimate_ate(
            sample_causal_data, "treatment", "outcome",
            confounders=["age", "income"], method="ipw",
        )
        assert isinstance(result, dict)
        assert "ate" in result

    def test_estimate_ate_empty_data(self, engine):
        result = engine.estimate_ate(
            pd.DataFrame(), "treatment", "outcome", method="diff_in_means"
        )
        assert "error" in result

    def test_estimate_ate_unknown_method(self, engine, sample_causal_data):
        result = engine.estimate_ate(
            sample_causal_data, "treatment", "outcome", method="unknown"
        )
        assert "error" in result

    def test_estimate_cate(self, engine, sample_causal_data):
        result = engine.estimate_cate(
            sample_causal_data, "treatment", "outcome",
            heterogeneity_vars=["city_tier"],
            confounders=["age", "income"],
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# ElasticityEstimator
# ---------------------------------------------------------------------------


class TestElasticityEstimator:
    """Tests for src.tools.subsidy.elasticity.ElasticityEstimator."""

    @pytest.fixture
    def estimator(self):
        from src.tools.subsidy.elasticity import ElasticityEstimator
        return ElasticityEstimator()

    @pytest.fixture
    def sample_elasticity_data(self):
        np.random.seed(42)
        n = 100
        price = np.random.uniform(10, 100, n)
        demand = 1000 / price + np.random.normal(0, 5, n)
        demand = np.maximum(demand, 1)
        return pd.DataFrame({
            "price": price,
            "demand": demand,
            "season": np.random.choice(["spring", "summer", "fall", "winter"], n),
        })

    def test_instantiate(self, estimator):
        assert estimator is not None

    def test_estimate_price_elasticity_returns_dict(self, estimator, sample_elasticity_data):
        result = estimator.estimate_price_elasticity(sample_elasticity_data)
        assert isinstance(result, dict)
        assert "elasticity" in result
        assert isinstance(result["elasticity"], float)

    def test_estimate_price_elasticity_has_ci(self, estimator, sample_elasticity_data):
        result = estimator.estimate_price_elasticity(sample_elasticity_data)
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_estimate_price_elasticity_empty_data(self, estimator):
        result = estimator.estimate_price_elasticity(pd.DataFrame())
        assert "error" in result

    def test_estimate_price_elasticity_missing_columns(self, estimator):
        result = estimator.estimate_price_elasticity(
            pd.DataFrame({"a": [1], "b": [2]}), price_col="missing"
        )
        assert "error" in result

    def test_estimate_price_elasticity_with_controls(self, estimator, sample_elasticity_data):
        result = estimator.estimate_price_elasticity(
            sample_elasticity_data, control_vars=["season"]
        )
        assert isinstance(result, dict)
        assert "elasticity" in result


# ---------------------------------------------------------------------------
# BudgetOptimizer
# ---------------------------------------------------------------------------


class TestBudgetOptimizer:
    """Tests for src.tools.subsidy.budget_optimizer.BudgetOptimizer."""

    @pytest.fixture
    def optimizer(self):
        from src.tools.subsidy.budget_optimizer import BudgetOptimizer
        return BudgetOptimizer()

    @pytest.fixture
    def sample_segments(self):
        return {"new_user": 5000, "active": 3000, "dormant": 2000}

    @pytest.fixture
    def sample_effects(self):
        return {
            "new_user": {"ate": 0.05, "base_conversion_rate": 0.08},
            "active": {"ate": 0.03, "base_conversion_rate": 0.12},
            "dormant": {"ate": 0.07, "base_conversion_rate": 0.04},
        }

    def test_instantiate(self, optimizer):
        assert optimizer is not None

    def test_optimize_allocation_returns_dict(self, optimizer, sample_segments, sample_effects):
        result = optimizer.optimize_allocation(
            user_segments=sample_segments,
            causal_effects=sample_effects,
            total_budget=50000,
        )
        assert isinstance(result, dict)
        assert "allocation" in result or "total_budget" in result

    def test_optimize_allocation_empty_segments(self, optimizer):
        result = optimizer.optimize_allocation({}, {}, 10000)
        assert "error" in result

    def test_optimize_allocation_zero_budget(self, optimizer, sample_segments, sample_effects):
        result = optimizer.optimize_allocation(
            user_segments=sample_segments,
            causal_effects=sample_effects,
            total_budget=0,
        )
        assert "error" in result

    def test_optimize_allocation_negative_budget(self, optimizer, sample_segments, sample_effects):
        result = optimizer.optimize_allocation(
            user_segments=sample_segments,
            causal_effects=sample_effects,
            total_budget=-100,
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# SubsidyAllocator
# ---------------------------------------------------------------------------


class TestSubsidyAllocator:
    """Tests for src.tools.subsidy.subsidy_allocator.SubsidyAllocator."""

    @pytest.fixture
    def allocator(self):
        from src.tools.subsidy.subsidy_allocator import SubsidyAllocator
        return SubsidyAllocator()

    @pytest.fixture
    def sample_causal_results(self):
        return {"ate": 0.05}

    @pytest.fixture
    def sample_elasticity_results(self):
        return {"segment_elasticities": [
            {"segment": "new_user", "elasticity": -1.5},
            {"segment": "active", "elasticity": -0.8},
        ]}

    @pytest.fixture
    def sample_budget_plan(self):
        return {
            "allocation": {
                "new_user": {"coupon_amount": 15, "user_count": 5000, "expected_incremental_orders": 250},
                "active": {"coupon_amount": 10, "user_count": 3000, "expected_incremental_orders": 90},
            },
            "total_budget": 105000,
        }

    def test_instantiate(self, allocator):
        assert allocator is not None

    def test_allocate_returns_dict(self, allocator, sample_causal_results, sample_elasticity_results, sample_budget_plan):
        result = allocator.allocate(
            sample_causal_results, sample_elasticity_results, sample_budget_plan
        )
        assert isinstance(result, dict)

    def test_allocate_with_error_budget(self, allocator, sample_causal_results, sample_elasticity_results):
        result = allocator.allocate(
            sample_causal_results,
            sample_elasticity_results,
            {"error": "optimization failed"},
        )
        assert "error" in result

    def test_allocate_empty_allocation(self, allocator, sample_causal_results, sample_elasticity_results):
        result = allocator.allocate(
            sample_causal_results,
            sample_elasticity_results,
            {"allocation": {}},
        )
        assert "error" in result

    def test_allocate_valid_plan_has_segments(self, allocator, sample_causal_results, sample_elasticity_results, sample_budget_plan):
        result = allocator.allocate(
            sample_causal_results, sample_elasticity_results, sample_budget_plan
        )
        # Should have segment details or summary
        assert isinstance(result, dict)
        assert result  # not empty
