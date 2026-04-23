"""ElasticityEstimator - estimate price elasticity of demand.

Computes overall and segment-level price elasticity using OLS regression,
with optional instrumental variable (IV/2SLS) support for endogeneity correction.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@ToolRegistry.register("elasticity")
class ElasticityEstimator:
    """Estimate price elasticity of demand (own-price elasticity)."""

    def estimate_price_elasticity(
        self,
        data: pd.DataFrame,
        price_col: str = "price",
        demand_col: str = "demand",
        instrument_vars: list[str] | None = None,
        control_vars: list[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate the price elasticity of demand.

        Uses log-log regression: ln(demand) = alpha + beta * ln(price) + controls
        where beta is the price elasticity.

        If instrument_vars are provided, uses 2SLS (Two-Stage Least Squares)
        to correct for endogeneity.

        Args:
            data: DataFrame with price and demand columns.
            price_col: column name for price.
            demand_col: column name for demand quantity.
            instrument_vars: optional list of instrument variable column names
                for 2SLS estimation.
            control_vars: optional list of control variable column names.

        Returns:
            Dict with elasticity estimate, confidence interval, and diagnostics.
        """
        if data.empty:
            return {"error": "data is empty"}

        missing = {price_col, demand_col} - set(data.columns)
        if missing:
            return {"error": f"missing columns: {missing}"}

        control_vars = control_vars or []
        instrument_vars = instrument_vars or []

        # Prepare data: drop rows with NaN/inf in key columns
        req_cols = [price_col, demand_col] + [
            c for c in control_vars if c in data.columns
        ]
        clean = data[req_cols].dropna().copy()

        # Filter positive values (log transform requires > 0)
        clean = clean[(clean[price_col] > 0) & (clean[demand_col] > 0)]
        if len(clean) < 10:
            return {"error": f"need at least 10 valid observations, got {len(clean)}"}

        clean["_ln_price"] = np.log(clean[price_col])
        clean["_ln_demand"] = np.log(clean[demand_col])

        if instrument_vars:
            available_iv = [v for v in instrument_vars if v in data.columns]
            if available_iv:
                clean_iv = data.loc[clean.index, available_iv].dropna()
                common_idx = clean.index.intersection(clean_iv.index)
                clean = clean.loc[common_idx]
                if len(clean) < 10:
                    return {"error": "insufficient data after filtering instrument variables"}
                return self._estimate_2sls(clean, price_col, demand_col, available_iv, control_vars)

        # Standard OLS log-log regression
        return self._estimate_ols(clean, price_col, demand_col, control_vars)

    def segment_elasticity(
        self,
        data: pd.DataFrame,
        segment_col: str = "segment",
        price_col: str = "price",
        demand_col: str = "demand",
        control_vars: list[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate price elasticity per segment.

        Args:
            data: DataFrame with price, demand, and segment columns.
            segment_col: column name for the segment variable.
            price_col: column name for price.
            demand_col: column name for demand.
            control_vars: optional control variables.

        Returns:
            Dict with per-segment elasticity estimates and overall comparison.
        """
        if data.empty:
            return {"error": "data is empty"}

        if segment_col not in data.columns:
            return {"error": f"column '{segment_col}' not found in data"}

        control_vars = control_vars or []
        results: dict[str, dict[str, Any]] = []

        for seg_val, seg_data in data.groupby(segment_col):
            est = self.estimate_price_elasticity(
                seg_data, price_col, demand_col, control_vars=control_vars
            )
            if "error" in est:
                est["segment"] = str(seg_val)
                est["n_obs"] = len(seg_data)
            else:
                est["segment"] = str(seg_val)
            results[str(seg_val)] = est

        # Overall elasticity for comparison
        overall = self.estimate_price_elasticity(
            data, price_col, demand_col, control_vars=control_vars
        )

        # Summarise
        segment_list: list[dict[str, Any]] = []
        for seg_name, est in results.items():
            segment_list.append(
                {
                    "segment": seg_name,
                    "elasticity": est.get("elasticity"),
                    "se": est.get("se"),
                    "ci_lower": est.get("ci_lower"),
                    "ci_upper": est.get("ci_upper"),
                    "r_squared": est.get("r_squared"),
                    "n_obs": est.get("n_obs", est.get("n_total", 0)),
                }
            )

        # Sort by absolute elasticity (most sensitive first)
        segment_list.sort(key=lambda x: abs(x.get("elasticity") or 0), reverse=True)

        return {
            "overall_elasticity": overall if "error" not in overall else None,
            "segment_elasticities": segment_list,
            "most_elastic_segment": segment_list[0]["segment"] if segment_list else None,
            "least_elastic_segment": segment_list[-1]["segment"] if segment_list else None,
            "heterogeneity_observed": (
                max(abs(s.get("elasticity") or 0) for s in segment_list)
                - min(abs(s.get("elasticity") or 0) for s in segment_list)
            )
            > 0.2
            if segment_list
            else False,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_ols(
        data: pd.DataFrame,
        price_col: str,
        demand_col: str,
        control_vars: list[str],
    ) -> dict[str, Any]:
        """OLS log-log regression for elasticity."""
        y = data["_ln_demand"].values
        X_parts: list[np.ndarray] = [np.ones(len(data))]
        X_parts.append(data["_ln_price"].values)

        for ctrl in control_vars:
            if ctrl in data.columns:
                vals = pd.to_numeric(data[ctrl], errors="coerce").fillna(0).values
                X_parts.append(vals)

        X = np.column_stack(X_parts)
        n, k = X.shape

        # OLS
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"error": "OLS regression failed"}

        y_pred = X @ beta
        residuals = y - y_pred
        sigma2 = float(np.sum(residuals**2) / max(n - k, 1))

        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.zeros(k)

        # Elasticity is the coefficient on ln(price) (index 1)
        elasticity = float(beta[1])
        elasticity_se = float(se[1])
        ci_lower = elasticity - 1.96 * elasticity_se
        ci_upper = elasticity + 1.96 * elasticity_se
        t_stat = elasticity / elasticity_se if elasticity_se > 0 else 0.0
        p_value = float(2 * stats.norm.sf(abs(t_stat)))

        # R-squared
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "method": "OLS_log_log",
            "elasticity": round(elasticity, 6),
            "se": round(elasticity_se, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant_at_05": p_value < 0.05,
            "r_squared": round(r_squared, 4),
            "interpretation": _interpret_elasticity(elasticity),
            "n_obs": n,
        }

    @staticmethod
    def _estimate_2sls(
        data: pd.DataFrame,
        price_col: str,
        demand_col: str,
        instruments: list[str],
        control_vars: list[str],
    ) -> dict[str, Any]:
        """Two-Stage Least Squares (2SLS) for IV estimation."""
        y = data["_ln_demand"].values

        # Stage 1: regress ln(price) on instruments + controls
        Z_parts: list[np.ndarray] = [np.ones(len(data))]
        for iv in instruments:
            vals = pd.to_numeric(data[iv], errors="coerce").fillna(0).values
            Z_parts.append(vals)
        for ctrl in control_vars:
            if ctrl in data.columns:
                vals = pd.to_numeric(data[ctrl], errors="coerce").fillna(0).values
                Z_parts.append(vals)
        Z = np.column_stack(Z_parts)

        endog = data["_ln_price"].values

        # First stage
        try:
            pi = np.linalg.lstsq(Z, endog, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"error": "First stage regression failed"}

        fitted_price = Z @ pi

        # First-stage F-statistic for instrument relevance
        ss_res_1 = float(np.sum((endog - fitted_price)**2))
        ss_tot_1 = float(np.sum((endog - np.mean(endog))**2))
        r2_first = 1 - ss_res_1 / ss_tot_1 if ss_tot_1 > 0 else 0.0
        n_instruments = len(instruments)
        n_obs = len(data)
        f_stat = (r2_first / n_instruments) / ((1 - r2_first) / max(n_obs - Z.shape[1], 1)) if (1 - r2_first) > 0 else 0.0

        # Stage 2: regress ln(demand) on fitted ln(price) + controls
        X2_parts: list[np.ndarray] = [np.ones(len(data)), fitted_price]
        for ctrl in control_vars:
            if ctrl in data.columns:
                vals = pd.to_numeric(data[ctrl], errors="coerce").fillna(0).values
                X2_parts.append(vals)
        X2 = np.column_stack(X2_parts)

        try:
            beta_2sls = np.linalg.lstsq(X2, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"error": "Second stage regression failed"}

        elasticity = float(beta_2sls[1])

        # Robust SE (simplified)
        residuals_2 = y - X2 @ beta_2sls
        sigma2_2 = float(np.sum(residuals_2**2) / max(n_obs - X2.shape[1], 1))

        try:
            cov_2 = sigma2_2 * np.linalg.inv(X2.T @ X2)
            se_2 = float(np.sqrt(cov_2[1, 1]))
        except np.linalg.LinAlgError:
            se_2 = 0.0

        ci_lower = elasticity - 1.96 * se_2
        ci_upper = elasticity + 1.96 * se_2
        t_stat = elasticity / se_2 if se_2 > 0 else 0.0
        p_value = float(2 * stats.norm.sf(abs(t_stat)))

        return {
            "method": "2SLS",
            "elasticity": round(elasticity, 6),
            "se": round(se_2, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant_at_05": p_value < 0.05,
            "first_stage_r_squared": round(r2_first, 4),
            "first_stage_f_stat": round(f_stat, 4),
            "instruments_weak": f_stat < 10,
            "interpretation": _interpret_elasticity(elasticity),
            "n_obs": n_obs,
        }


def _interpret_elasticity(e: float) -> str:
    """Return a human-readable interpretation of elasticity."""
    e = abs(e)
    if e < 0.5:
        return "inelastic (|e| < 0.5): demand is relatively insensitive to price"
    elif e < 1.0:
        return "relatively inelastic (0.5 <= |e| < 1.0): demand responds moderately to price"
    elif e < 1.5:
        return "unit elastic to elastic (1.0 <= |e| < 1.5): demand is sensitive to price"
    elif e < 2.0:
        return "elastic (1.5 <= |e| < 2.0): demand is quite sensitive to price"
    else:
        return "highly elastic (|e| >= 2.0): demand is very sensitive to price"
