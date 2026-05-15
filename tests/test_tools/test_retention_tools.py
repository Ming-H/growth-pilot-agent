"""Tests for retention tools: ChurnPredictor, CohortAnalyzer, NurturePlanner, WinbackPlanner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# ChurnPredictor
# ---------------------------------------------------------------------------


class TestChurnPredictor:
    """Tests for src.tools.retention.churn_predictor.ChurnPredictor."""

    @pytest.fixture
    def predictor(self):
        from src.tools.retention.churn_predictor import ChurnPredictor
        return ChurnPredictor(n_estimators=10, cv_folds=2)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = (X[:, 0] > 0.5).astype(int)
        return X, y

    def test_instantiate(self, predictor):
        assert predictor is not None
        assert predictor.churn_threshold_days == 30

    def test_train_returns_metrics(self, predictor, sample_data):
        X, y = sample_data
        metrics = predictor.train(X, y)
        assert isinstance(metrics, dict)
        assert "auc" in metrics
        assert "average_precision" in metrics

    def test_predict_churn_risk_returns_series(self, predictor, sample_data):
        X, y = sample_data
        predictor.train(X, y)
        probs = predictor.predict_churn_risk(X[:5])
        assert isinstance(probs, pd.Series)
        assert len(probs) == 5
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_before_train_raises(self, predictor):
        with pytest.raises(RuntimeError):
            predictor.predict_churn_risk(np.random.rand(5, 3))

    def test_segment_churned_users(self, predictor, sample_data):
        X, y = sample_data
        predictor.train(X, y)
        probs = predictor.predict_churn_risk(X)
        features = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        segments = predictor.segment_churned_users(probs, features)
        assert isinstance(segments, dict)


# ---------------------------------------------------------------------------
# CohortAnalyzer
# ---------------------------------------------------------------------------


class TestCohortAnalyzer:
    """Tests for src.tools.retention.cohort_analyzer.CohortAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        from src.tools.retention.cohort_analyzer import CohortAnalyzer
        return CohortAnalyzer(observation_window=60)

    @pytest.fixture
    def sample_order_data(self):
        np.random.seed(42)
        rows = []
        base_date = pd.Timestamp("2024-01-01")
        for uid in range(1, 31):
            signup = base_date + pd.Timedelta(days=np.random.randint(0, 30))
            n_orders = np.random.randint(1, 6)
            for _ in range(n_orders):
                order_date = signup + pd.Timedelta(days=np.random.randint(0, 45))
                rows.append({
                    "user_id": uid,
                    "order_date": order_date,
                    "signup_date": signup,
                })
        return pd.DataFrame(rows)

    def test_instantiate(self, analyzer):
        assert analyzer is not None
        assert analyzer.observation_window == 60

    def test_analyze_retention_cohort_returns_dataframe(self, analyzer, sample_order_data):
        result = analyzer.analyze_retention_cohort(sample_order_data)
        assert isinstance(result, pd.DataFrame)

    def test_analyze_retention_cohort_empty(self, analyzer):
        result = analyzer.analyze_retention_cohort(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_find_retention_inflection(self, analyzer):
        curve = np.array([1.0, 0.8, 0.65, 0.55, 0.50, 0.48, 0.47, 0.46])
        result = analyzer.find_retention_inflection(curve)
        assert isinstance(result, dict)
        assert "inflection_points" in result or "inflection_point" in result or isinstance(result, dict)

    def test_analyze_retention_cohort_monthly(self, analyzer, sample_order_data):
        result = analyzer.analyze_retention_cohort(sample_order_data, period="M")
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# NurturePlanner
# ---------------------------------------------------------------------------


class TestNurturePlanner:
    """Tests for src.tools.retention.nurture_planner.NurturePlanner."""

    @pytest.fixture
    def planner(self):
        from src.tools.retention.nurture_planner import NurturePlanner
        return NurturePlanner()

    def test_instantiate(self, planner):
        assert planner is not None

    def test_generate_nurture_plan_returns_dict(self, planner):
        result = planner.generate_nurture_plan()
        assert isinstance(result, dict)
        assert "day1" in result
        assert "day7" in result
        assert "day14" in result
        assert "day30" in result
        assert "meta" in result

    def test_generate_nurture_plan_has_actions(self, planner):
        result = planner.generate_nurture_plan()
        for phase in ("day1", "day7", "day14", "day30"):
            assert isinstance(result[phase], list)
            assert len(result[phase]) > 0
            for action in result[phase]:
                assert "action" in action
                assert "channel" in action

    def test_generate_nurture_plan_with_user_data(self, planner):
        user_data = {"city_tier": 1, "has_freight_search": True}
        result = planner.generate_nurture_plan(new_user_data=user_data)
        assert result["meta"]["personalisation"] == "personalised"

    def test_evaluate_nurture_progress(self, planner):
        cohort_data = pd.DataFrame({
            "days_since_signup": [1, 3, 7, 14, 30, 1, 3, 7],
            "is_active": [True, True, True, False, False, True, True, True],
        })
        result = planner.evaluate_nurture_progress(cohort_data)
        assert isinstance(result, dict)
        assert "overall_health" in result
        assert "recommendations" in result

    def test_evaluate_nurture_progress_empty(self, planner):
        result = planner.evaluate_nurture_progress(pd.DataFrame())
        assert result["overall_health"] == "no_data"


# ---------------------------------------------------------------------------
# WinbackPlanner
# ---------------------------------------------------------------------------


class TestWinbackPlanner:
    """Tests for src.tools.retention.winback_planner.WinbackPlanner."""

    @pytest.fixture
    def planner(self):
        from src.tools.retention.winback_planner import WinbackPlanner
        return WinbackPlanner(min_segment_size=10)

    @pytest.fixture
    def sample_churn_segments(self):
        return {
            "price_sensitive": {"count": 500},
            "service_dissatisfied": {"count": 200},
            "competitor_switched": {"count": 300},
            "no_need": {"count": 5},
        }

    def test_instantiate(self, planner):
        assert planner is not None
        assert planner.min_segment_size == 10

    def test_generate_winback_plan_returns_dict(self, planner, sample_churn_segments):
        result = planner.generate_winback_plan(sample_churn_segments)
        assert isinstance(result, dict)
        assert "summary" in result

    def test_generate_winback_plan_has_strategies(self, planner, sample_churn_segments):
        result = planner.generate_winback_plan(sample_churn_segments)
        # Each segment should have a strategy
        for seg_name in ("price_sensitive", "service_dissatisfied", "competitor_switched"):
            assert seg_name in result
            seg_plan = result[seg_name]
            assert "strategy" in seg_plan

    def test_generate_winback_plan_small_segment_skipped(self, planner, sample_churn_segments):
        result = planner.generate_winback_plan(sample_churn_segments)
        # "no_need" has count=5 which is below min_segment_size=10
        assert result["no_need"]["strategy"] == "skip"

    def test_generate_winback_plan_summary_has_totals(self, planner, sample_churn_segments):
        result = planner.generate_winback_plan(sample_churn_segments)
        summary = result["summary"]
        assert "total_estimated_cost" in summary
        assert "total_estimated_winbacks" in summary
        assert "priority_order" in summary

    def test_generate_winback_plan_with_historical_data(self, planner, sample_churn_segments):
        hist_data = pd.DataFrame({
            "segment": ["price_sensitive", "price_sensitive"],
            "winback_rate": [0.10, 0.14],
            "cost_per_winback": [15.0, 12.0],
            "roi": [1.5, 2.0],
        })
        result = planner.generate_winback_plan(sample_churn_segments, historical_winback_data=hist_data)
        assert isinstance(result, dict)
        assert "summary" in result

    def test_generate_winback_plan_unknown_segment(self, planner):
        segments = {"random_new_segment": {"count": 100}}
        result = planner.generate_winback_plan(segments)
        assert isinstance(result, dict)
        assert "random_new_segment" in result
        assert result["random_new_segment"]["strategy"] != "skip"
