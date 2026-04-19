"""Freight intent prediction model using LightGBM.

Trains a binary classifier that predicts whether a ride-hailing user is
likely to convert to freight/transport services, based on engineered features.
Includes probability calibration via IsotonicRegression.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class IntentModel:
    """LightGBM-based freight intent prediction model.

    Typical workflow::

        model = IntentModel()
        metrics = model.train(X_train, y_train)
        probs = model.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        cv_folds: int = 5,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.cv_folds = cv_folds

        self._model: lgb.LGBMClassifier | None = None
        self._calibrator: IsotonicRegression | None = None
        self._feature_names: list[str] | None = None
        self._is_calibrated: bool = False

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Train the LightGBM classifier with cross-validation.

        Returns a dict with keys:
            auc, accuracy, cv_auc_mean, cv_auc_std, feature_importance
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        if len(X_arr) < 10:
            logger.warning("Training data too small (%d rows), using minimal config", len(X_arr))
            self.n_estimators = 50
            self.cv_folds = min(self.cv_folds, max(2, len(X_arr) // 3))

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        # LightGBM classifier
        self._model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=min(self.min_child_samples, max(1, len(X_arr) // 10)),
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbose=-1,
        )

        # Cross-validation AUC
        cv_scores: list[float] = []
        actual_folds = min(self.cv_folds, len(np.unique(y_arr)))
        if actual_folds >= 2:
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
        y_pred_label = (y_pred_prob >= 0.5).astype(int)

        try:
            train_auc = float(roc_auc_score(y_arr, y_pred_prob))
        except ValueError:
            train_auc = 0.5

        train_acc = float(accuracy_score(y_arr, y_pred_label))

        # Feature importance
        fi = self._get_raw_feature_importance()

        result: dict[str, Any] = {
            "auc": train_auc,
            "accuracy": train_acc,
            "cv_auc_mean": float(np.mean(cv_scores)) if cv_scores else None,
            "cv_auc_std": float(np.std(cv_scores)) if cv_scores else None,
            "feature_importance": fi,
        }

        logger.info(
            "IntentModel trained – AUC=%.4f Acc=%.4f CV-AUC=%.4f+/-%.4f",
            train_auc,
            train_acc,
            result["cv_auc_mean"] or 0,
            result["cv_auc_std"] or 0,
        )
        return result

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame | np.ndarray) -> pd.Series:
        """Return predicted freight-intent probabilities for each row."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_arr = np.asarray(X)
        probs = self._model.predict_proba(X_arr)[:, 1]

        if self._is_calibrated and self._calibrator is not None:
            probs = self._calibrator.transform(probs)

        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.Series(probs, index=index, name="intent_prob")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of features ranked by importance (gain-based)."""
        fi = self._get_raw_feature_importance()
        return pd.DataFrame(
            list(fi.items()),
            columns=["feature", "importance"],
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    def _get_raw_feature_importance(self) -> dict[str, float]:
        """Get feature importance dict (gain), normalised to sum to 1."""
        if self._model is None:
            return {}

        gains = self._model.booster_.feature_importance(importance_type="gain")
        names = self._model.booster_.feature_name()

        if self._feature_names is not None and len(names) == len(self._feature_names):
            names = self._feature_names

        total = gains.sum()
        if total == 0:
            total = 1.0

        return {str(n): float(g / total) for n, g in zip(names, gains)}

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> None:
        """Fit an isotonic-regression calibrator on held-out predictions.

        After calibration, :meth:`predict` will return calibrated probabilities.
        """
        y_true_arr = np.asarray(y_true).ravel()
        y_prob_arr = np.asarray(y_prob).ravel()

        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(y_prob_arr, y_true_arr)
        self._is_calibrated = True
        logger.info("IntentModel calibrated with isotonic regression (%d samples)", len(y_true_arr))

    # ------------------------------------------------------------------
    # Sample data for testing
    # ------------------------------------------------------------------

    @staticmethod
    def generate_sample_data(
        n_samples: int = 2000,
        n_features: int = 20,
        positive_rate: float = 0.08,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic training data for testing.

        Returns (X, y) where y has roughly *positive_rate* positives.
        """
        rng = np.random.RandomState(seed)
        X = pd.DataFrame(
            rng.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Make some features predictive of the target
        logits = (
            0.5 * X["feature_0"]
            + 0.3 * X["feature_1"]
            + 0.2 * X["feature_2"]
            + rng.logistic(size=n_samples)
        )
        # Shift threshold to achieve desired positive rate
        threshold = np.quantile(logits, 1 - positive_rate)
        y = (logits >= threshold).astype(int)

        return X, pd.Series(y, name="freight_intent")
