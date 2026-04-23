"""Customer lifetime-value (LTV) prediction module.

Supports two approaches:
1. **Probabilistic**: BG/NBD + Gamma-Gamma model from the ``lifetimes`` library.
2. **ML-based**: LightGBM regression on engineered features.

Also computes LTV/CAC ratios by acquisition channel.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@ToolRegistry.register("ltv_predictor")
class LVTPredictor:
    """Predict customer lifetime value and analyse LTV/CAC ratios.

    Typical usage (ML mode)::

        pred = LVTPredictor(method="ml")
        pred.train(user_features, historical_ltv)
        ltv = pred.predict_ltv(user_features, cohort_history)

    Or (probabilistic mode)::

        pred = LVTPredictor(method="probabilistic")
        ltv = pred.predict_ltv(cohort_data, cohort_data)
    """

    def __init__(
        self,
        method: str = "ml",
        prediction_horizon: int = 180,
        discount_rate: float = 0.01,
        random_state: int = 42,
    ) -> None:
        """Initialise predictor.

        Parameters
        ----------
        method : str
            ``"ml"`` for LightGBM regression, ``"probabilistic"`` for
            BG/NBD + Gamma-Gamma.
        prediction_horizon : int
            Number of days ahead to predict (used by probabilistic model).
        discount_rate : float
            Daily discount rate for monetary value estimation.
        random_state : int
            Random seed.
        """
        if method not in ("ml", "probabilistic"):
            raise ValueError(f"method must be 'ml' or 'probabilistic', got '{method}'")
        self.method = method
        self.prediction_horizon = prediction_horizon
        self.discount_rate = discount_rate
        self.random_state = random_state

        self._reg_model: lgb.LGBMRegressor | None = None
        self._bgf: Any = None  # BetaGeoFitter from lifetimes
        self._ggf: Any = None  # GammaGammaFitter from lifetimes
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Train (ML mode)
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Train the LightGBM regression model (only for ML method).

        Parameters
        ----------
        X : feature matrix
        y : historical LTV values

        Returns dict with ``rmse``, ``mae``, ``r2`` on training data.
        """
        if self.method != "ml":
            raise RuntimeError("train() is only for method='ml'")

        X_arr = np.asarray(X)
        y_arr = np.asarray(y).ravel()

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        self._reg_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1,
        )
        self._reg_model.fit(X_arr, y_arr)

        # Training metrics
        y_pred = self._reg_model.predict(X_arr)
        residuals = y_arr - y_pred
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        logger.info("LVTPredictor trained – RMSE=%.2f MAE=%.2f R2=%.4f", rmse, mae, r2)
        return metrics

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_ltv(
        self,
        user_features: pd.DataFrame | np.ndarray,
        cohort_history: pd.DataFrame | np.ndarray | None = None,
    ) -> pd.Series:
        """Predict LTV for each user.

        For *ML mode*: uses the trained LightGBM model on *user_features*.
        For *probabilistic mode*: uses BG/NBD + Gamma-Gamma on
        *cohort_history* (expected columns: ``customer_id``, ``frequency``,
        ``recency``, ``T``, ``monetary_value``).

        Returns a Series of predicted LTV values indexed by user ID.
        """
        if self.method == "ml":
            return self._predict_ml(user_features)
        else:
            return self._predict_probabilistic(
                cohort_history if cohort_history is not None else user_features
            )

    def _predict_ml(self, user_features: pd.DataFrame | np.ndarray) -> pd.Series:
        """Predict using trained LightGBM regressor."""
        if self._reg_model is None:
            raise RuntimeError("ML model not trained. Call train() first.")

        X_arr = np.asarray(user_features)
        preds = self._reg_model.predict(X_arr)
        preds = np.clip(preds, 0, None)  # LTV cannot be negative

        index = user_features.index if isinstance(user_features, pd.DataFrame) else None
        return pd.Series(preds, index=index, name="predicted_ltv")

    def _predict_probabilistic(self, cohort_data: pd.DataFrame | np.ndarray) -> pd.Series:
        """Predict using BG/NBD + Gamma-Gamma (lifetimes library)."""
        df = cohort_data.copy() if isinstance(cohort_data, pd.DataFrame) else pd.DataFrame(np.asarray(cohort_data))

        required = {"customer_id", "frequency", "recency", "T", "monetary_value"}
        missing = required - set(df.columns)
        if missing:
            # Try to build lifetimes-compatible columns from simpler input
            df = self._prepare_lifetimes_data(df)

        try:
            from lifetimes import BetaGeoFitter, GammaGammaFitter

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._bgf = BetaGeoFitter(penalizer_coef=0.0)
                self._bgf.fit(
                    df["frequency"],
                    df["recency"],
                    df["T"],
                )

                self._ggf = GammaGammaFitter(penalizer_coef=0.0)
                self._ggf.fit(
                    df["frequency"],
                    df["monetary_value"],
                )

                # Predict conditional expected number of purchases
                predicted_purchases = self._bgf.conditional_expected_number_of_purchases_up_to_time(
                    self.prediction_horizon,
                    df["frequency"],
                    df["recency"],
                    df["T"],
                )

                # Predict expected average transaction value
                predicted_monetary = self._ggf.conditional_expected_average_profit(
                    df["frequency"],
                    df["monetary_value"],
                )

                ltv = predicted_purchases * predicted_monetary

            return pd.Series(
                np.asarray(ltv),
                index=df["customer_id"].values if "customer_id" in df.columns else None,
                name="predicted_ltv",
            )

        except ImportError:
            logger.warning("lifetimes not installed, falling back to heuristic LTV")
            return self._heuristic_ltv(df)

    # ------------------------------------------------------------------
    # Channel analysis
    # ------------------------------------------------------------------

    def ltv_by_channel(
        self,
        conversion_data: pd.DataFrame,
        ltv_predictions: pd.Series | np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Compute average LTV by acquisition channel.

        Parameters
        ----------
        conversion_data : pd.DataFrame
            Must contain ``user_id`` and ``channel`` columns.
        ltv_predictions : array-like
            Predicted LTV, indexed by user_id.

        Returns
        -------
        dict
            ``{channel: {"mean_ltv": ..., "median_ltv": ..., "count": ...}}``
        """
        ltv_series = pd.Series(np.asarray(ltv_predictions).ravel(), name="ltv")

        if isinstance(conversion_data, pd.DataFrame) and "channel" in conversion_data.columns:
            merged = conversion_data[["user_id", "channel"]].copy()
            merged["ltv"] = ltv_series.values[: len(merged)]
        else:
            return {"unknown": {"mean_ltv": float(ltv_series.mean()), "count": len(ltv_series)}}

        result: dict[str, dict[str, float]] = {}
        for ch, grp in merged.groupby("channel"):
            result[str(ch)] = {
                "mean_ltv": float(grp["ltv"].mean()),
                "median_ltv": float(grp["ltv"].median()),
                "count": int(len(grp)),
            }
        return result

    def compute_ltv_cac_ratio(
        self,
        ltv_predictions: pd.Series | np.ndarray,
        cac_by_channel: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Compute LTV / CAC ratio by channel.

        Parameters
        ----------
        ltv_predictions : array-like
            Predicted LTV per user.
        cac_by_channel : dict
            ``{channel: customer_acquisition_cost}``

        Returns
        -------
        dict
            ``{channel: {"mean_ltv": ..., "cac": ..., "ltv_cac_ratio": ...}}``
        """
        ltv_arr = np.asarray(ltv_predictions).ravel()
        mean_ltv = float(np.mean(ltv_arr))

        result: dict[str, dict[str, float]] = {}
        for ch, cac in cac_by_channel.items():
            # Use overall mean LTV if no channel-specific LTV is available
            result[ch] = {
                "mean_ltv": mean_ltv,
                "cac": float(cac),
                "ltv_cac_ratio": mean_ltv / float(cac) if cac > 0 else float("inf"),
            }
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_lifetimes_data(df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to create lifetimes-compatible columns from simpler data."""
        out = df.copy()

        if "customer_id" not in out.columns and "user_id" in out.columns:
            out["customer_id"] = out["user_id"]

        if "frequency" not in out.columns and "ride_count" in out.columns:
            out["frequency"] = out["ride_count"]

        if "recency" not in out.columns and "days_since_last_ride" in out.columns:
            if "days_since_signup" in out.columns:
                out["recency"] = out["days_since_signup"] - out["days_since_last_ride"]
            else:
                out["recency"] = 30 - out["days_since_last_ride"].clip(upper=30)

        if "T" not in out.columns and "days_since_signup" in out.columns:
            out["T"] = out["days_since_signup"]

        for col, default in [
            ("frequency", 1),
            ("recency", 15),
            ("T", 30),
            ("monetary_value", 50.0),
            ("customer_id", range(len(out))),
        ]:
            if col not in out.columns:
                if isinstance(default, (int, float)):
                    out[col] = default
                else:
                    out[col] = list(default)

        return out

    @staticmethod
    def _heuristic_ltv(df: pd.DataFrame) -> pd.Series:
        """Fallback heuristic: aov * frequency * retention_factor."""
        freq = df.get("frequency", pd.Series([1] * len(df)))
        mv = df.get("monetary_value", pd.Series([50.0] * len(df)))
        ltv = np.asarray(freq) * np.asarray(mv) * 1.2  # retention factor
        idx = df["customer_id"].values if "customer_id" in df.columns else None
        return pd.Series(ltv, index=idx, name="predicted_ltv")
