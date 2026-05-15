"""Tests for conversion tools: FunnelAnalyzer, ReachPlanner, SlotAllocator, CouponDesigner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# FunnelAnalyzer
# ---------------------------------------------------------------------------


class TestFunnelAnalyzer:
    """Tests for src.tools.conversion.funnel_analyzer.FunnelAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        from src.tools.conversion.funnel_analyzer import FunnelAnalyzer
        return FunnelAnalyzer()

    @pytest.fixture
    def sample_funnel_dict(self):
        return {
            "exposure": 10000,
            "click": 2000,
            "app_open": 1500,
            "search": 800,
            "quote_view": 600,
            "order_confirm": 400,
            "first_order": 300,
        }

    def test_instantiate(self, analyzer):
        assert analyzer is not None

    def test_analyze_funnel_returns_dict(self, analyzer, sample_funnel_dict):
        result = analyzer.analyze_funnel(sample_funnel_dict)
        assert isinstance(result, dict)
        assert "stages" in result
        assert "overall_conversion_rate" in result
        assert "bottleneck" in result
        assert result["total_users_entered"] == 10000
        assert result["total_users_converted"] == 300

    def test_analyze_funnel_with_dataframe(self, analyzer):
        df = pd.DataFrame({
            "stage": ["exposure", "click", "order"],
            "count": [1000, 300, 50],
        })
        result = analyzer.analyze_funnel(df, stages=["exposure", "click", "order"])
        assert isinstance(result, dict)
        assert result["total_users_entered"] == 1000

    def test_analyze_funnel_identifies_bottleneck(self, analyzer, sample_funnel_dict):
        result = analyzer.analyze_funnel(sample_funnel_dict)
        bottleneck = result["bottleneck"]
        assert "stage" in bottleneck
        assert "stage_conversion_rate" in bottleneck
        assert bottleneck["stage_conversion_rate"] <= 1.0

    def test_analyze_funnel_empty_dict(self, analyzer):
        result = analyzer.analyze_funnel({})
        assert "error" in result

    def test_analyze_funnel_single_stage(self, analyzer):
        result = analyzer.analyze_funnel({"exposure": 1000})
        assert "error" in result

    def test_bottleneck_diagnosis(self, analyzer):
        segment_data = pd.DataFrame({
            "segment": ["organic", "paid", "social"],
            "count_before": [500, 300, 200],
            "count_after": [100, 30, 60],
        })
        result = analyzer.bottleneck_diagnosis("click", segment_data)
        assert isinstance(result, dict)

    def test_bottleneck_diagnosis_empty(self, analyzer):
        result = analyzer.bottleneck_diagnosis("click", pd.DataFrame())
        assert "error" in result


# ---------------------------------------------------------------------------
# ReachPlanner
# ---------------------------------------------------------------------------


class TestReachPlanner:
    """Tests for src.tools.conversion.reach_planner.ReachPlanner."""

    @pytest.fixture
    def planner(self):
        from src.tools.conversion.reach_planner import ReachPlanner
        return ReachPlanner()

    @pytest.fixture
    def sample_segments(self):
        return {"new_user": 5000, "dormant": 2000, "active": 8000}

    def test_instantiate(self, planner):
        assert planner is not None

    def test_plan_reach_strategy_returns_dict(self, planner, sample_segments):
        result = planner.plan_reach_strategy(sample_segments)
        assert isinstance(result, dict)

    def test_plan_reach_strategy_empty_segments(self, planner):
        result = planner.plan_reach_strategy({})
        assert "error" in result

    def test_plan_reach_strategy_with_constraints(self, planner, sample_segments):
        constraints = {
            "budget": 500.0,
            "max_daily_push": 10000,
            "blacklist_channels": ["SMS"],
        }
        result = planner.plan_reach_strategy(sample_segments, constraints=constraints)
        assert isinstance(result, dict)

    def test_plan_reach_strategy_with_channel_performance(self, planner, sample_segments):
        channel_perf = {
            "Push": {"ctr": 0.08, "conversion_rate": 0.02},
        }
        result = planner.plan_reach_strategy(sample_segments, channel_performance=channel_perf)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# SlotAllocator
# ---------------------------------------------------------------------------


class TestSlotAllocator:
    """Tests for src.tools.conversion.slot_allocator.SlotAllocator."""

    @pytest.fixture
    def allocator(self):
        from src.tools.conversion.slot_allocator import SlotAllocator
        return SlotAllocator()

    @pytest.fixture
    def sample_user_segments(self):
        return {
            "high_value": {"count": 500, "ltv": 500, "priority": 5},
            "new_user": {"count": 3000, "ltv": 50, "priority": 3},
            "dormant": {"count": 2000, "ltv": 80, "priority": 2},
        }

    def test_instantiate(self, allocator):
        assert allocator is not None

    def test_allocate_slots_returns_dict(self, allocator, sample_user_segments):
        result = allocator.allocate_slots(sample_user_segments)
        assert isinstance(result, dict)

    def test_allocate_slots_empty_segments(self, allocator):
        result = allocator.allocate_slots({})
        assert "error" in result

    def test_allocate_slots_with_custom_capacity(self, allocator, sample_user_segments):
        capacity = {"slot_a": 3, "slot_b": 2}
        result = allocator.allocate_slots(sample_user_segments, slot_capacity=capacity)
        assert isinstance(result, dict)

    def test_allocate_slots_with_performance_data(self, allocator, sample_user_segments):
        perf = {
            "金刚位": {
                "high_value": {"ctr": 0.1, "conversion_rate": 0.05},
                "new_user": {"ctr": 0.05, "conversion_rate": 0.02},
            },
        }
        result = allocator.allocate_slots(
            sample_user_segments, performance_data=perf
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# CouponDesigner
# ---------------------------------------------------------------------------


class TestCouponDesigner:
    """Tests for src.tools.conversion.coupon_designer.CouponDesigner."""

    @pytest.fixture
    def designer(self):
        from src.tools.conversion.coupon_designer import CouponDesigner
        return CouponDesigner()

    def test_instantiate(self, designer):
        assert designer is not None

    def test_design_coupon_returns_dict(self, designer):
        result = designer.design_coupon("new_user")
        assert isinstance(result, dict)
        assert "coupon_type" in result
        assert "segment" in result
        assert result["segment"] == "new_user"

    def test_design_coupon_has_estimated_rates(self, designer):
        result = designer.design_coupon("dormant")
        assert "estimated_claim_rate" in result
        assert "estimated_redemption_rate" in result
        assert 0 <= result["estimated_claim_rate"] <= 1
        assert 0 <= result["estimated_redemption_rate"] <= 1

    def test_design_coupon_with_budget_constraint(self, designer):
        result = designer.design_coupon("active", budget_constraint=10.0)
        assert isinstance(result, dict)
        assert "coupon_type" in result

    def test_design_coupon_unknown_segment_uses_defaults(self, designer):
        result = designer.design_coupon("unknown_segment_xyz")
        assert isinstance(result, dict)
        assert "coupon_type" in result

    def test_compare_coupon_types(self, designer):
        result = designer.compare_coupon_types("new_user")
        assert isinstance(result, dict)
        assert "recommendation" in result

    def test_design_coupon_with_history(self, designer):
        history = pd.DataFrame({
            "coupon_type": ["折扣券", "满减券"],
            "amount": [15, 10],
            "threshold": [0, 30],
            "claim_rate": [0.4, 0.6],
            "redemption_rate": [0.2, 0.3],
            "orders_generated": [50, 80],
        })
        result = designer.design_coupon("active", historical_coupon_data=history)
        assert isinstance(result, dict)
        assert "coupon_type" in result
