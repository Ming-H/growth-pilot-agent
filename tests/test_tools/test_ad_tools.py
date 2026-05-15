"""Tests for ad tools: RTAStrategy, BidOptimizer, CreativeAnalyzer, AudienceAnalyzer."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# RTAStrategy
# ---------------------------------------------------------------------------


class TestRTAStrategy:
    """Tests for src.tools.ad.rta_strategy.RTAStrategy."""

    @pytest.fixture
    def strategy(self):
        from src.tools.ad.rta_strategy import RTAStrategy
        return RTAStrategy()

    @pytest.fixture
    def sample_user_features(self):
        return {
            "age": 30,
            "city_tier": 1,
            "historical_orders": 5,
            "days_since_last_order": 3,
            "device_type": "ios",
        }

    @pytest.fixture
    def sample_prospect_scores(self):
        return {
            "conversion_prob": 0.08,
            "churn_risk": 0.2,
            "ltv_estimate": 300,
            "intent_score": 0.7,
        }

    def test_instantiate(self, strategy):
        assert strategy is not None

    def test_should_bid_returns_tuple(self, strategy, sample_user_features, sample_prospect_scores):
        result = strategy.should_bid(
            sample_user_features, sample_prospect_scores,
            bid_floor=0.5, target_cpa=50.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_should_bid_zero_bid_floor(self, strategy, sample_user_features, sample_prospect_scores):
        should, price = strategy.should_bid(
            sample_user_features, sample_prospect_scores,
            bid_floor=0, target_cpa=50.0,
        )
        assert should is False
        assert price == 0.0

    def test_should_bid_zero_target_cpa(self, strategy, sample_user_features, sample_prospect_scores):
        should, price = strategy.should_bid(
            sample_user_features, sample_prospect_scores,
            bid_floor=0.5, target_cpa=0,
        )
        assert should is False

    def test_should_bid_high_floor_may_reject(self, strategy, sample_user_features, sample_prospect_scores):
        # Very high floor should be rejected
        should, price = strategy.should_bid(
            sample_user_features, sample_prospect_scores,
            bid_floor=10000.0, target_cpa=5.0,
        )
        assert should is False

    def test_build_rta_decision_rules(self, strategy):
        historical = [
            {"user_features": {"city_tier": 1}, "outcome": "win", "cpa": 30, "revenue": 150},
            {"user_features": {"city_tier": 2}, "outcome": "loss", "cpa": None, "revenue": 0},
            {"user_features": {"city_tier": 1}, "outcome": "win", "cpa": 25, "revenue": 120},
        ]
        result = strategy.build_rta_decision_rules(historical)
        assert isinstance(result, dict)
        assert "rules" in result
        assert "segments" in result
        assert "overall_metrics" in result

    def test_build_rta_decision_rules_empty(self, strategy):
        result = strategy.build_rta_decision_rules([])
        assert result["rules"] == []
        assert result["segments"] == {}


# ---------------------------------------------------------------------------
# BidOptimizer
# ---------------------------------------------------------------------------


class TestBidOptimizer:
    """Tests for src.tools.ad.bid_optimizer.BidOptimizer."""

    @pytest.fixture
    def optimizer(self):
        from src.tools.ad.bid_optimizer import BidOptimizer
        return BidOptimizer()

    def test_instantiate(self, optimizer):
        assert optimizer is not None

    def test_ecpc_bid_returns_float(self, optimizer):
        bid = optimizer.ecpc_bid(
            current_bid=2.0, cvr=0.05, target_cpa=50.0, alpha=0.5
        )
        assert isinstance(bid, float)
        assert bid > 0

    def test_ecpc_bid_zero_target_cpa(self, optimizer):
        bid = optimizer.ecpc_bid(current_bid=2.0, cvr=0.05, target_cpa=0)
        assert bid == 0.0

    def test_ecpc_bid_zero_cvr(self, optimizer):
        bid = optimizer.ecpc_bid(current_bid=2.0, cvr=0, target_cpa=50.0)
        assert bid == 0.0

    def test_ecpc_bid_negative_current_bid(self, optimizer):
        bid = optimizer.ecpc_bid(current_bid=-1.0, cvr=0.05, target_cpa=50.0)
        assert bid == 0.0

    def test_ecpc_bid_reasonable_range(self, optimizer):
        bid = optimizer.ecpc_bid(
            current_bid=2.0, cvr=0.05, target_cpa=40.0, alpha=1.0
        )
        # bid should be formulaic: 2.0 * 0.05 / 40.0 = 0.0025, clamped to 0.01
        assert bid >= 0.01

    def test_pid_controller_returns_float(self, optimizer):
        errors = [5.0, 3.0, 1.0]
        output = optimizer.pid_controller(errors)
        assert isinstance(output, float)

    def test_pid_controller_empty_errors(self, optimizer):
        output = optimizer.pid_controller([])
        assert output == 0.0

    def test_pid_controller_single_error(self, optimizer):
        output = optimizer.pid_controller([5.0])
        assert isinstance(output, float)
        # P term: 0.3 * 5.0 = 1.5, I term: 0.1 * 5.0 = 0.5, total = 2.0
        assert output == pytest.approx(2.0, abs=0.01)

    def test_bid_simulation_returns_dict(self, optimizer):
        ctr_curve = {
            "bid_multipliers": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            "ctr_values": [0.01, 0.015, 0.02, 0.023, 0.025, 0.026],
        }
        cvr_curve = {
            "bid_multipliers": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
            "cvr_values": [0.02, 0.03, 0.04, 0.045, 0.047, 0.048],
        }
        result = optimizer.bid_simulation(
            budget=10000, base_bid=2.0,
            ctr_curve=ctr_curve, cvr_curve=cvr_curve,
        )
        assert isinstance(result, dict)
        assert "bid_levels" in result
        assert "results" in result
        assert "optimal_bid" in result
        assert "recommendation" in result
        assert len(result["results"]) == 6

    def test_bid_simulation_zero_budget(self, optimizer):
        result = optimizer.bid_simulation(budget=0, base_bid=2.0, ctr_curve={}, cvr_curve={})
        assert result["bid_levels"] == []
        assert result["optimal_bid"] == 0.0


# ---------------------------------------------------------------------------
# CreativeAnalyzer
# ---------------------------------------------------------------------------


class TestCreativeAnalyzer:
    """Tests for src.tools.ad.creative_analyzer.CreativeAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        from src.tools.ad.creative_analyzer import CreativeAnalyzer
        return CreativeAnalyzer()

    @pytest.fixture
    def sample_creative_data(self):
        return [
            {"creative_id": "c1", "impressions": 10000, "clicks": 500, "conversions": 25, "spend": 200, "revenue": 1000},
            {"creative_id": "c2", "impressions": 8000, "clicks": 200, "conversions": 10, "spend": 150, "revenue": 400},
            {"creative_id": "c3", "impressions": 5000, "clicks": 300, "conversions": 20, "spend": 120, "revenue": 800},
        ]

    def test_instantiate(self, analyzer):
        assert analyzer is not None

    def test_analyze_creative_performance_returns_dict(self, analyzer, sample_creative_data):
        result = analyzer.analyze_creative_performance(sample_creative_data)
        assert isinstance(result, dict)
        assert "creatives" in result
        assert "summary" in result
        assert "top_performers" in result
        assert "underperformers" in result

    def test_analyze_creative_performance_metrics(self, analyzer, sample_creative_data):
        result = analyzer.analyze_creative_performance(sample_creative_data)
        for c in result["creatives"]:
            assert "ctr" in c
            assert "cvr" in c
            assert "cpc" in c
            assert "cpa" in c
            assert "roi" in c

    def test_analyze_creative_performance_empty(self, analyzer):
        result = analyzer.analyze_creative_performance([])
        assert result["creatives"] == []

    def test_analyze_creative_performance_sorted_by_roi(self, analyzer, sample_creative_data):
        result = analyzer.analyze_creative_performance(sample_creative_data)
        rois = [c["roi"] for c in result["creatives"]]
        assert rois == sorted(rois, reverse=True)

    def test_detect_creative_fatigue(self, analyzer):
        # Simulate declining CTR over time
        time_series_data = [
            {"date": "2024-01-01", "ctr": 0.05},
            {"date": "2024-01-02", "ctr": 0.048},
            {"date": "2024-01-03", "ctr": 0.045},
            {"date": "2024-01-04", "ctr": 0.030},
            {"date": "2024-01-05", "ctr": 0.020},
        ]
        if hasattr(analyzer, "detect_creative_fatigue"):
            result = analyzer.detect_creative_fatigue(time_series_data)
            assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# AudienceAnalyzer
# ---------------------------------------------------------------------------


class TestAudienceAnalyzer:
    """Tests for src.tools.ad.audience_analyzer.AudienceAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        from src.tools.ad.audience_analyzer import AudienceAnalyzer
        return AudienceAnalyzer()

    @pytest.fixture
    def sample_audience_data(self):
        np.random.seed(42)
        return [
            {
                "user_id": f"u{i}",
                "age": np.random.randint(20, 55),
                "gender": np.random.choice(["M", "F"]),
                "city_tier": np.random.randint(1, 5),
                "historical_orders": np.random.randint(0, 30),
                "avg_order_value": round(np.random.uniform(20, 200), 2),
                "days_since_last_order": np.random.randint(1, 90),
                "ltv": round(np.random.uniform(0, 1000), 2),
            }
            for i in range(50)
        ]

    def test_instantiate(self, analyzer):
        assert analyzer is not None

    def test_analyze_audience_returns_dict(self, analyzer, sample_audience_data):
        result = analyzer.analyze_audience(sample_audience_data)
        assert isinstance(result, dict)
        assert "total_users" in result
        assert result["total_users"] == 50
        assert "segments" in result
        assert "demographics" in result
        assert "behavioral_summary" in result

    def test_analyze_audience_empty(self, analyzer):
        result = analyzer.analyze_audience([])
        assert result["total_users"] == 0
        assert result["segments"] == {}

    def test_analyze_audience_demographics(self, analyzer, sample_audience_data):
        result = analyzer.analyze_audience(sample_audience_data)
        demo = result["demographics"]
        assert "age_distribution" in demo
        assert "age_mean" in demo
        assert "gender_distribution" in demo

    def test_analyze_audience_behavioral_summary(self, analyzer, sample_audience_data):
        result = analyzer.analyze_audience(sample_audience_data)
        bs = result["behavioral_summary"]
        assert "avg_orders_per_user" in bs
        assert "avg_ltv" in bs
        assert "avg_recency_days" in bs

    def test_lookalike_expansion(self, analyzer, sample_audience_data):
        seed_users = sample_audience_data[:5]
        all_users = sample_audience_data
        result = analyzer.lookalike_expansion(seed_users, all_users, top_k=10)
        assert isinstance(result, list)
        assert len(result) <= 10
        # Seed users should not be in results
        seed_ids = {u["user_id"] for u in seed_users}
        for uid in result:
            assert uid not in seed_ids

    def test_lookalike_expansion_empty_seed(self, analyzer):
        result = analyzer.lookalike_expansion([], [{"user_id": "u1"}])
        assert result == []

    def test_lookalike_expansion_empty_pool(self, analyzer):
        result = analyzer.lookalike_expansion([{"user_id": "s1"}], [])
        assert result == []
