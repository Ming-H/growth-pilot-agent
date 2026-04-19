"""Churn prediction model using LightGBM.

Predicts the probability that an active user will churn (no ride in
the next 30 days) and segments churned users by behavioural characteristics.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """LightGBM-based churn prediction for ride-hailing users.

    Typical workflow::

        cp = ChurnPredictor()
        metrics = cp.train(X_train, y_train)
        churn_probs = cp.predict_churn_risk(X_test)
        segments = cp.segment_churned_users(churn_probs, user_features)
    """

    def __init__(
        self,
        churn_threshold_days: int = 30,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float | None = None,
        random_state: int = 42,
        cv_folds: int = 5,
    ) -> None:
        self.churn_threshold_days = churn_threshold_days
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.cv_folds = cv_folds

        self._model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Train the churn prediction model.

        Parameters
        ----------
        X : feature matrix (n_samples, n_features)
        y : binary labels (1 = churned, 0 = retained)

        Returns
        -------
        dict
            Keys: ``auc``, ``average_precision``, ``cv_auc_mean``,
            ``cv_auc_std``, ``feature_importance``
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        # Handle class imbalance with scale_pos_weight if not specified
        pos_count = y_arr.sum()
        neg_count = len(y_arr) - pos_count
        spw = self.scale_pos_weight
        if spw is None and pos_count > 0 and neg_count > 0:
            spw = neg_count / pos_count

        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=min(self.min_child_samples, max(1, len(X_arr) // 10)),
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=spw,
            random_state=self.random_state,
            verbose=-1,
        )

        # Cross-validation
        cv_scores: list[float] = []
        actual_folds = min(self.cv_folds, len(np.unique(y_arr)))
        if actual_folds >= 2 and len(X_arr) >= 20:
            skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=self.random_state)
            for train_idx, val_idx in skf.split(X_arr, y_arr):
                X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
                clone = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    num_leaves=self.num_leaves,
                    min_child_samples=min(self.min_child_samples, max(1, len(X_tr) // 10)),
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    scale_pos_weight=spw,
                    random_state=self.random_state,
                    verbose=-1,
                )
                clone.fit(X_tr, y_tr)
                y_pred_val = clone.predict_proba(X_val)[:, 1]
                try:
                    cv_scores.append(roc_auc_score(y_val, y_pred_val))
                except ValueError:
                    pass

        # Fit on full data
        self._model.fit(X_arr, y_arr)

        # Training metrics
        y_pred_prob = self._model.predict_proba(X_arr)[:, 1]
        try:
            train_auc = float(roc_auc_score(y_arr, y_pred_prob))
        except ValueError:
            train_auc = 0.5

        try:
            train_ap = float(average_precision_score(y_arr, y_pred_prob))
        except ValueError:
            train_ap = 0.0

        # Feature importance
        fi = self._get_feature_importance()

        result: dict[str, Any] = {
            "auc": train_auc,
            "average_precision": train_ap,
            "cv_auc_mean": float(np.mean(cv_scores)) if cv_scores else None,
            "cv_auc_std": float(np.std(cv_scores)) if cv_scores else None,
            "feature_importance": fi,
        }

        logger.info(
            "ChurnPredictor trained – AUC=%.4f AP=%.4f CV=%.4f+/-%.4f",
            train_auc,
            train_ap,
            result["cv_auc_mean"] or 0,
            result["cv_auc_std"] or 0,
        )
        return result

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_churn_risk(self, X: pd.DataFrame | np.ndarray) -> pd.Series:
        """Predict churn probability for each user.

        Returns a Series of churn probabilities indexed like *X*.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_arr = np.asarray(X)
        probs = self._model.predict_proba(X_arr)[:, 1]

        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.Series(probs, index=index, name="churn_prob")

    # ------------------------------------------------------------------
    # Segment churned users
    # ------------------------------------------------------------------

    def segment_churned_users(
        self,
        churn_scores: pd.Series | np.ndarray,
        user_features: pd.DataFrame,
        high_threshold: float = 0.7,
        medium_threshold: float = 0.4,
    ) -> dict[str, Any]:
        """Segment users by churn risk level and behavioural patterns.

        Parameters
        ----------
        churn_scores : array-like
            Churn probabilities from :meth:`predict_churn_risk`.
        user_features : pd.DataFrame
            Must contain ``user_id`` column.
        high_threshold : float
            Score above which users are classified as high-risk.
        medium_threshold : float
            Score above which users are classified as medium-risk.

        Returns
        -------
        dict
            Keys:
                ``high_risk``, ``medium_risk``, ``low_risk`` – each a
                dict with ``count``, ``avg_churn_score``, and behavioural
                profile summaries.
        """
        probs = np.asarray(churn_scores).ravel()
        df = user_features.copy()

        if len(probs) != len(df):
            logger.warning(
                "Length mismatch between churn_scores (%d) and user_features (%d). "
                "Using minimum length.",
                len(probs),
                len(df),
            )
            n = min(len(probs), len(df))
            probs = probs[:n]
            df = df.iloc[:n]

        df["churn_prob"] = probs

        segments: dict[str, Any] = {}

        for label, mask in [
            ("high_risk", df["churn_prob"] >= high_threshold),
            ("medium_risk", (df["churn_prob"] >= medium_threshold) & (df["churn_prob"] < high_threshold)),
            ("low_risk", df["churn_prob"] < medium_threshold),
        ]:
            seg_df = df[mask]
            profile = self._summarise_segment(seg_df)
            profile["count"] = int(len(seg_df))
            profile["avg_churn_score"] = float(seg_df["churn_prob"].mean()) if len(seg_df) > 0 else 0.0
            segments[label] = profile

        # Overall summary
        segments["summary"] = {
            "total_users": int(len(df)),
            "high_risk_pct": float((df["churn_prob"] >= high_threshold).mean()),
            "medium_risk_pct": float(((df["churn_prob"] >= medium_threshold) & (df["churn_prob"] < high_threshold)).mean()),
            "low_risk_pct": float((df["churn_prob"] < medium_threshold).mean()),
        }

        return segments

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        fi = self._get_feature_importance()
        return pd.DataFrame(
            list(fi.items()),
            columns=["feature", "importance"],
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    def _get_feature_importance(self) -> dict[str, float]:
        """Get normalised gain-based feature importance."""
        if self._model is None:
            return {}
        gains = self._model.booster_.feature_importance(importance_type="gain")
        names = self._model.booster_.feature_name()
        if self._feature_names and len(names) == len(self._feature_names):
            names = self._feature_names
        total = gains.sum()
        if total == 0:
            total = 1.0
        return {str(n): float(g / total) for n, g in zip(names, gains)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_segment(seg_df: pd.DataFrame) -> dict[str, Any]:
        """Summarise key behavioural stats for a user segment."""
        if seg_df.empty:
            return {}

        summary: dict[str, Any] = {}

        # Numeric column summaries
        for col in [
            "ride_count_30d", "avg_ride_distance", "night_ride_ratio",
            "weekend_ride_ratio", "active_days_30d",
        ]:
            if col in seg_df.columns:
                summary[f"mean_{col}"] = float(seg_df[col].mean())

        # City tier distribution
        if "city_tier" in seg_df.columns:
            summary["city_tier_dist"] = seg_df["city_tier"].value_counts(normalize=True).to_dict()

        # Freight search rate
        if "has_freight_search" in seg_df.columns:
            summary["freight_search_rate"] = float(seg_df["has_freight_search"].mean())

        return summary

    # ------------------------------------------------------------------
    # Sample data
    # ------------------------------------------------------------------

    @staticmethod
    def generate_sample_data(
        n_samples: int = 3000,
        n_features: int = 15,
        churn_rate: float = 0.25,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic churn data for testing."""
        rng = np.random.RandomState(seed)

        X = pd.DataFrame(
            rng.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Make features predictive of churn
        logits = (
            -0.8 * X["feature_0"]   # negative coefficient = higher feature_0 = less churn
            - 0.4 * X["feature_1"]
            + 0.3 * X["feature_2"]
            + rng.logistic(size=n_samples)
        )
        threshold = np.quantile(logits, 1 - churn_rate)
        y = (logits >= threshold).astype(int)  # high logit = churn

        return X, pd.Series(y, name="churned")
