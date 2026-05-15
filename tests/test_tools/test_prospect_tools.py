"""Tests for prospect tools: FeatureEngine, IntentModel, UserScorer, LTVPredictor, UserSegmentor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# FeatureEngine
# ---------------------------------------------------------------------------


class TestFeatureEngine:
    """Tests for src.tools.prospect.feature_engine.FeatureEngine."""

    @pytest.fixture
    def engine(self):
        from src.tools.prospect.feature_engine import FeatureEngine
        return FeatureEngine()

    @pytest.fixture
    def sample_logs(self):
        """Create minimal ride-hailing log DataFrame."""
        now = pd.Timestamp.now()
        rows = []
        for uid in range(1, 6):
            for i in range(3):
                rows.append({
                    "user_id": uid,
                    "ride_time": now - pd.Timedelta(days=i * 3),
                    "distance": 5.0 + uid + i,
                    "destination_type": "residential",
                    "fare": 15.0 + uid,
                    "ride_duration": 20.0 + uid,
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_profiles(self):
        return pd.DataFrame({
            "user_id": range(1, 6),
            "city_tier": [1, 2, 3, 1, 2],
            "has_freight_search": [True, False, True, False, True],
            "has_large_item_search": [False, True, False, True, False],
        })

    def test_instantiate(self, engine):
        assert engine is not None

    def test_extract_behavior_features_returns_dataframe(self, engine, sample_logs):
        result = engine.extract_behavior_features(sample_logs)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_extract_behavior_features_empty_input(self, engine):
        result = engine.extract_behavior_features(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_extract_temporal_features_returns_dataframe(self, engine, sample_logs):
        result = engine.extract_temporal_features(sample_logs)
        assert isinstance(result, pd.DataFrame)

    def test_extract_temporal_features_empty_input(self, engine):
        result = engine.extract_temporal_features(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_extract_context_features_returns_dataframe(self, engine, sample_profiles):
        result = engine.extract_context_features(sample_profiles)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_extract_context_features_empty_input(self, engine):
        result = engine.extract_context_features(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_build_feature_matrix(self, engine, sample_logs, sample_profiles):
        raw_data = {"user_logs": sample_logs, "user_profile": sample_profiles}
        result = engine.build_feature_matrix(raw_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] > 0


# ---------------------------------------------------------------------------
# IntentModel
# ---------------------------------------------------------------------------


class TestIntentModel:
    """Tests for src.tools.prospect.intent_model.IntentModel."""

    @pytest.fixture
    def model(self):
        from src.tools.prospect.intent_model import IntentModel
        return IntentModel(n_estimators=10, cv_folds=2)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.rand(50, 5)
        y = (X[:, 0] > 0.5).astype(int)
        return X, y

    def test_instantiate(self, model):
        assert model is not None
        assert model.n_estimators == 10

    def test_train_returns_metrics(self, model, sample_data):
        X, y = sample_data
        metrics = model.train(X, y)
        assert isinstance(metrics, dict)
        assert "auc" in metrics
        assert "feature_importance" in metrics

    def test_predict_returns_series(self, model, sample_data):
        X, y = sample_data
        model.train(X, y)
        probs = model.predict(X[:5])
        assert isinstance(probs, pd.Series)
        assert len(probs) == 5
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_before_train_raises(self, model):
        X = np.random.rand(5, 3)
        with pytest.raises(RuntimeError):
            model.predict(X)


# ---------------------------------------------------------------------------
# UserScorer
# ---------------------------------------------------------------------------


class TestUserScorer:
    """Tests for src.tools.prospect.user_scorer.UserScorer."""

    @pytest.fixture
    def scorer(self):
        from src.tools.prospect.user_scorer import UserScorer
        return UserScorer()

    def test_instantiate(self, scorer):
        assert scorer is not None
        assert scorer.intent_weight == 0.6
        assert scorer.ltv_weight == 0.4

    def test_score_users_returns_dataframe(self, scorer):
        intent = np.array([0.8, 0.3, 0.6])
        ltv = np.array([200, 50, 150])
        result = scorer.score_users(intent, ltv)
        assert isinstance(result, pd.DataFrame)
        assert "composite_score" in result.columns
        assert "intent_score" in result.columns
        assert "ltv_norm" in result.columns
        assert len(result) == 3

    def test_score_users_with_ids(self, scorer):
        intent = np.array([0.9, 0.5])
        ltv = np.array([300, 100])
        ids = pd.Series(["u1", "u2"])
        result = scorer.score_users(intent, ltv, user_ids=ids)
        assert list(result.index) == ["u1", "u2"]

    def test_score_users_length_mismatch_raises(self, scorer):
        intent = np.array([0.5, 0.6])
        ltv = np.array([100])
        with pytest.raises(ValueError, match="Length mismatch"):
            scorer.score_users(intent, ltv)

    def test_score_users_single_value(self, scorer):
        intent = np.array([0.5])
        ltv = np.array([100])
        result = scorer.score_users(intent, ltv)
        assert len(result) == 1
        assert result["composite_score"].iloc[0] == pytest.approx(0.5 * 0.6 + 0.0 * 0.4)

    def test_rank_users(self, scorer):
        intent = np.array([0.8, 0.3, 0.6])
        ltv = np.array([200, 50, 150])
        scored = scorer.score_users(intent, ltv)
        ranked = scorer.rank_users(scored)
        assert "rank" in ranked.columns
        assert ranked["rank"].iloc[0] == 1

    def test_rank_users_empty(self, scorer):
        ranked = scorer.rank_users(pd.DataFrame())
        assert ranked.empty

    def test_segment_by_score(self, scorer):
        intent = np.array([0.9, 0.6, 0.3, 0.1])
        ltv = np.array([400, 200, 100, 50])
        scored = scorer.score_users(intent, ltv)
        segments = scorer.segment_by_score(scored)
        assert isinstance(segments, dict)
        assert "high_intent" in segments
        assert "cold" in segments

    def test_segment_by_score_empty(self, scorer):
        segments = scorer.segment_by_score(pd.DataFrame())
        assert isinstance(segments, dict)
        assert "high_intent" in segments


# ---------------------------------------------------------------------------
# LTVPredictor
# ---------------------------------------------------------------------------


class TestLTVPredictor:
    """Tests for src.tools.prospect.ltv_predictor.LTVPredictor."""

    def test_instantiate_ml(self):
        from src.tools.prospect.ltv_predictor import LTVPredictor
        pred = LTVPredictor(method="ml")
        assert pred.method == "ml"

    def test_instantiate_probabilistic(self):
        from src.tools.prospect.ltv_predictor import LTVPredictor
        pred = LTVPredictor(method="probabilistic")
        assert pred.method == "probabilistic"

    def test_instantiate_invalid_method_raises(self):
        from src.tools.prospect.ltv_predictor import LTVPredictor
        with pytest.raises(ValueError, match="method must be"):
            LTVPredictor(method="invalid")

    def test_train_and_predict_ml(self):
        from src.tools.prospect.ltv_predictor import LTVPredictor
        pred = LTVPredictor(method="ml", random_state=42)
        np.random.seed(42)
        X = pd.DataFrame({
            "ride_count": np.random.randint(1, 50, 40),
            "avg_fare": np.random.uniform(10, 100, 40),
            "recency_days": np.random.randint(1, 60, 40),
        })
        y = X["ride_count"] * X["avg_fare"] * 0.5 + np.random.normal(0, 50, 40)
        metrics = pred.train(X, y)
        assert isinstance(metrics, dict)
        assert "rmse" in metrics

        preds = pred.predict_ltv(X.head(5))
        assert isinstance(preds, pd.Series)
        assert len(preds) == 5

    def test_compute_ltv_cac_ratio(self):
        from src.tools.prospect.ltv_predictor import LTVPredictor
        pred = LTVPredictor(method="ml")
        ltv_predictions = pd.Series([300, 200, 150, 250])
        cac_by_channel = {"organic": 50, "paid_search": 80, "social": 60}
        result = pred.compute_ltv_cac_ratio(ltv_predictions, cac_by_channel)
        assert isinstance(result, dict)
        assert "organic" in result
        assert "ltv_cac_ratio" in result["organic"]


# ---------------------------------------------------------------------------
# UserSegmentor
# ---------------------------------------------------------------------------


class TestUserSegmentor:
    """Tests for src.tools.prospect.segmentor.UserSegmentor."""

    @pytest.fixture
    def segmentor(self):
        from src.tools.prospect.segmentor import UserSegmentor
        return UserSegmentor()

    @pytest.fixture
    def sample_user_data(self):
        now = pd.Timestamp.now()
        return pd.DataFrame({
            "user_id": range(1, 11),
            "last_ride_date": [now - pd.Timedelta(days=d) for d in [1, 3, 7, 14, 30, 5, 10, 60, 90, 2]],
            "ride_count": [20, 15, 8, 3, 1, 12, 6, 2, 0, 25],
            "total_spent": [2000, 1500, 500, 100, 30, 1200, 400, 80, 0, 3000],
        })

    def test_instantiate(self, segmentor):
        assert segmentor is not None
        assert segmentor.n_bins == 5

    def test_rfm_segmentation_returns_dataframe(self, segmentor, sample_user_data):
        result = segmentor.rfm_segmentation(sample_user_data)
        assert isinstance(result, pd.DataFrame)
        assert "rfm_score" in result.columns
        assert "rfm_segment" in result.columns
        assert len(result) == 10

    def test_rfm_segmentation_empty(self, segmentor):
        result = segmentor.rfm_segmentation(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_combined_segmentation(self, segmentor, sample_user_data):
        result = segmentor.combined_segmentation(sample_user_data)
        assert isinstance(result, pd.DataFrame)
        assert "lifecycle_stage" in result.columns
        assert len(result) == 10

    def test_combined_segmentation_empty(self, segmentor):
        result = segmentor.combined_segmentation(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty
