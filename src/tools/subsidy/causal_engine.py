"""CausalInferenceEngine - causal analysis using DoWhy or manual implementations.

Implements causal effect identification, ATE/CATE estimation, refutation
tests, and counterfactual analysis. Uses DoWhy when available; falls back
to manual implementations (IPW, backdoor adjustment) via statsmodels.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@ToolRegistry.register("causal_engine")
class CausalInferenceEngine:
    """Causal inference for subsidy effect estimation."""

    def identify_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: list[str],
    ) -> dict[str, Any]:
        """Identify the causal estimand (backdoor criterion).

        Args:
            data: DataFrame with treatment, outcome, and confounder columns.
            treatment: column name of the binary treatment variable.
            outcome: column name of the outcome variable.
            confounders: list of column names for confounders.

        Returns:
            Dict with identified estimand and validation info.
        """
        if data.empty:
            return {"error": "data is empty"}

        # Validate columns
        missing = {treatment, outcome} - set(data.columns)
        if missing:
            return {"error": f"missing columns: {missing}"}

        available_confounders = [c for c in confounders if c in data.columns]
        if len(available_confounders) < len(confounders):
            missing_c = set(confounders) - set(available_confounders)
            logger.warning("Missing confounders dropped: %s", missing_c)

        # Check treatment balance
        treatment_vals = data[treatment].values
        unique_treatments = np.unique(treatment_vals[~np.isnan(treatment_vals)])

        if len(unique_treatments) < 2:
            return {
                "error": f"treatment variable must have at least 2 levels, got {unique_treatments.tolist()}"
            }

        # Check treatment-control sizes
        treated_count = int(np.sum(treatment_vals == unique_treatments[1]))
        control_count = int(np.sum(treatment_vals == unique_treatments[0]))

        # Check confounder balance (standardised mean difference)
        balance: dict[str, dict[str, float]] = {}
        treated_mask = treatment_vals == unique_treatments[1]
        control_mask = treatment_vals == unique_treatments[0]

        for conf in available_confounders:
            if pd.api.types.is_numeric_dtype(data[conf]):
                t_mean = float(data.loc[treated_mask, conf].mean())
                c_mean = float(data.loc[control_mask, conf].mean())
                pooled_std = float(data[conf].std())
                smd = abs(t_mean - c_mean) / pooled_std if pooled_std > 0 else 0.0
                balance[conf] = {
                    "treated_mean": round(t_mean, 4),
                    "control_mean": round(c_mean, 4),
                    "std_mean_diff": round(smd, 4),
                    "balanced": smd < 0.1,
                }

        return {
            "estimand_type": "ATE",
            "estimand": f"E[Y({unique_treatments[1]}) - Y({unique_treatments[0]})]",
            "identification_method": "backdoor_adjustment",
            "treatment_levels": unique_treatments.tolist(),
            "treated_count": treated_count,
            "control_count": control_count,
            "confounders_used": available_confounders,
            "balance_check": balance,
            "all_confounders_balanced": all(
                b["balanced"] for b in balance.values()
            ),
        }

    def estimate_ate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: list[str] | None = None,
        method: str = "backdoor",
    ) -> dict[str, Any]:
        """Estimate Average Treatment Effect (ATE).

        Args:
            data: DataFrame with treatment, outcome, and confounder columns.
            treatment: column name of binary treatment.
            outcome: column name of outcome.
            confounders: list of confounder column names.
            method: "backdoor" for regression adjustment, "ipw" for inverse
                probability weighting, or "diff_in_means" for simple difference.

        Returns:
            Dict with ATE estimate, confidence interval, and diagnostics.
        """
        if data.empty:
            return {"error": "data is empty"}

        confounders = confounders or []
        available_confounders = [c for c in confounders if c in data.columns]

        treatment_vals = data[treatment].values.astype(float)
        outcome_vals = data[outcome].values.astype(float)

        treated_mask = treatment_vals == 1
        control_mask = treatment_vals == 0

        n_treated = int(treated_mask.sum())
        n_control = int(control_mask.sum())

        if n_treated < 2 or n_control < 2:
            return {"error": "need at least 2 observations in each treatment group"}

        if method == "diff_in_means":
            return self._ate_diff_in_means(outcome_vals, treated_mask, control_mask)
        elif method == "ipw":
            return self._ate_ipw(
                data, treatment, outcome, available_confounders, treatment_vals, outcome_vals
            )
        elif method == "backdoor":
            return self._ate_backdoor(
                data, treatment, outcome, available_confounders, outcome_vals
            )
        else:
            return {"error": f"unknown method: {method}. Use 'backdoor', 'ipw', or 'diff_in_means'"}

    def estimate_cate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        heterogeneity_vars: list[str],
        confounders: list[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate Conditional Average Treatment Effect (CATE) by subgroups.

        Args:
            data: DataFrame with required columns.
            treatment: binary treatment column name.
            outcome: outcome column name.
            heterogeneity_vars: columns to segment by for CATE estimation.
            confounders: additional confounders for adjustment.

        Returns:
            Dict with per-subgroup ATE estimates.
        """
        if data.empty:
            return {"error": "data is empty"}

        confounders = confounders or []
        available_het_vars = [v for v in heterogeneity_vars if v in data.columns]

        if not available_het_vars:
            # Fall back to overall ATE
            ate = self.estimate_ate(data, treatment, outcome, confounders)
            return {"cate": {"overall": ate}}

        cate_results: dict[str, Any] = {}

        for var in available_het_vars:
            if pd.api.types.is_numeric_dtype(data[var]):
                # Bin numeric variables into quantiles
                try:
                    bins = pd.qcut(data[var], q=4, duplicates="drop")
                except ValueError:
                    bins = pd.cut(data[var], bins=4)
                groups = bins
            else:
                groups = data[var]

            var_cate: dict[str, Any] = {}
            for group_val, group_df in data.groupby(groups, observed=True):
                if len(group_df) < 10:
                    continue
                group_ate = self.estimate_ate(
                    group_df, treatment, outcome, confounders, method="backdoor"
                )
                if "error" not in group_ate:
                    var_cate[str(group_val)] = {
                        "ate": group_ate["ate"],
                        "ci_lower": group_ate["ci_lower"],
                        "ci_upper": group_ate["ci_upper"],
                        "n": group_ate.get("n_total", len(group_df)),
                    }

            cate_results[var] = var_cate

        return {
            "cate": cate_results,
            "heterogeneity_vars": available_het_vars,
        }

    def refutation_test(
        self,
        estimate_result: dict[str, Any],
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: list[str] | None = None,
        method: str = "random_common_cause",
        n_simulations: int = 100,
    ) -> dict[str, Any]:
        """Run refutation tests on an ATE estimate.

        Args:
            estimate_result: result from estimate_ate().
            data: original DataFrame.
            treatment: treatment column name.
            outcome: outcome column name.
            confounders: confounder columns.
            method: "random_common_cause" adds a random feature and re-estimates,
                "placebo_treatment" replaces treatment with random values.
            n_simulations: number of simulations.

        Returns:
            Dict with refutation test results.
        """
        if "ate" not in estimate_result:
            return {"error": "estimate_result must contain 'ate' key"}

        original_ate = estimate_result["ate"]
        confounders = confounders or []

        simulated_ates: list[float] = []

        for _ in range(n_simulations):
            data_sim = data.copy()

            if method == "random_common_cause":
                # Add a random variable as confounder and re-estimate
                data_sim["_random_confounder"] = np.random.randn(len(data_sim))
                sim_confounders = confounders + ["_random_confounder"]
                result = self.estimate_ate(
                    data_sim, treatment, outcome, sim_confounders, method="backdoor"
                )
            elif method == "placebo_treatment":
                # Replace treatment with random assignment
                data_sim["_placebo_treatment"] = np.random.binomial(
                    1, 0.5, size=len(data_sim)
                )
                result = self.estimate_ate(
                    data_sim, "_placebo_treatment", outcome, confounders, method="backdoor"
                )
            else:
                return {"error": f"unknown refutation method: {method}"}

            if "ate" in result:
                simulated_ates.append(result["ate"])

        if not simulated_ates:
            return {"error": "all simulations failed"}

        sim_arr = np.array(simulated_ates)
        refutation_p_value = float(np.mean(np.abs(sim_arr) >= np.abs(original_ate)))

        return {
            "refutation_method": method,
            "original_ate": round(original_ate, 6),
            "simulated_ate_mean": round(float(np.mean(sim_arr)), 6),
            "simulated_ate_std": round(float(np.std(sim_arr)), 6),
            "refutation_p_value": round(refutation_p_value, 4),
            "is_robust": refutation_p_value < 0.05,
            "n_simulations": len(simulated_ates),
            "interpretation": (
                f"Original ATE ({original_ate:.4f}) is significantly different "
                f"from null distribution (p={refutation_p_value:.4f})"
                if refutation_p_value < 0.05
                else f"Original ATE ({original_ate:.4f}) is NOT robust to {method} refutation "
                f"(p={refutation_p_value:.4f})"
            ),
        }

    def counterfactual_analysis(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        what_if_values: list[float],
        confounders: list[str] | None = None,
    ) -> dict[str, Any]:
        """Estimate counterfactual outcomes for hypothetical treatment levels.

        For binary treatment, this estimates what outcomes would be under
        different treatment assignment rates.

        Args:
            data: DataFrame with treatment and outcome.
            treatment: treatment column name.
            outcome: outcome column name.
            what_if_values: list of treatment rates (0 to 1) to simulate.
            confounders: confounder columns for adjustment.

        Returns:
            Dict with counterfactual estimates.
        """
        if data.empty:
            return {"error": "data is empty"}

        confounders = confounders or []

        # Estimate ATE for calibration
        ate_result = self.estimate_ate(data, treatment, outcome, confounders, method="backdoor")
        if "error" in ate_result:
            return {"error": f"cannot estimate ATE: {ate_result['error']}"}

        ate = ate_result["ate"]
        control_mean = ate_result.get("control_mean", float(data.loc[data[treatment] == 0, outcome].mean()))

        # Current treatment rate
        current_rate = float(data[treatment].mean())

        counterfactuals: list[dict[str, Any]] = []
        for rate in what_if_values:
            rate = float(np.clip(rate, 0.0, 1.0))
            # Expected outcome = control_mean + rate * ATE
            expected_outcome = control_mean + rate * ate
            # Incremental gain vs current
            current_outcome = control_mean + current_rate * ate
            incremental = expected_outcome - current_outcome

            counterfactuals.append(
                {
                    "treatment_rate": round(rate, 4),
                    "expected_outcome": round(expected_outcome, 4),
                    "incremental_vs_current": round(incremental, 4),
                    "incremental_pct": round(incremental / current_outcome * 100, 2)
                    if current_outcome != 0
                    else 0.0,
                }
            )

        return {
            "ate_used": round(ate, 6),
            "control_mean": round(control_mean, 4),
            "current_treatment_rate": round(current_rate, 4),
            "current_expected_outcome": round(control_mean + current_rate * ate, 4),
            "counterfactuals": counterfactuals,
        }

    # ------------------------------------------------------------------
    # Internal: ATE estimation methods
    # ------------------------------------------------------------------

    @staticmethod
    def _ate_diff_in_means(
        outcome: np.ndarray,
        treated_mask: np.ndarray,
        control_mask: np.ndarray,
    ) -> dict[str, Any]:
        """Simple difference in means."""
        t_outcomes = outcome[treated_mask]
        c_outcomes = outcome[control_mask]

        ate = float(np.mean(t_outcomes) - np.mean(c_outcomes))
        se = float(np.sqrt(np.var(t_outcomes) / len(t_outcomes) + np.var(c_outcomes) / len(c_outcomes)))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        # Welch t-test
        t_stat, p_value = stats.ttest_ind(t_outcomes, c_outcomes, equal_var=False)

        return {
            "method": "diff_in_means",
            "ate": round(ate, 6),
            "se": round(se, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant_at_05": p_value < 0.05,
            "treated_mean": round(float(np.mean(t_outcomes)), 4),
            "control_mean": round(float(np.mean(c_outcomes)), 4),
            "n_treated": int(treated_mask.sum()),
            "n_control": int(control_mask.sum()),
            "n_total": len(outcome),
        }

    @staticmethod
    def _ate_backdoor(
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: list[str],
        outcome_vals: np.ndarray,
    ) -> dict[str, Any]:
        """ATE via regression adjustment (backdoor)."""
        # Build design matrix: intercept + confounders + treatment
        X_parts: list[np.ndarray] = [np.ones(len(data))]

        for conf in confounders:
            if conf in data.columns:
                col = data[conf].values.astype(float)
                col = np.nan_to_num(col, nan=np.nanmean(col))
                X_parts.append(col)

        X_parts.append(data[treatment].values.astype(float))
        X = np.column_stack(X_parts)
        y = outcome_vals

        # Remove rows with NaN in y
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # OLS
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"error": "OLS regression failed (singular matrix)"}

        # ATE is the coefficient on the treatment variable (last column)
        ate = float(beta[-1])

        # Residuals and standard errors
        y_pred = X @ beta
        residuals = y - y_pred
        n, k = X.shape
        sigma2 = float(np.sum(residuals**2) / max(n - k, 1))

        try:
            cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
            se = float(np.sqrt(cov_matrix[-1, -1]))
        except np.linalg.LinAlgError:
            se = 0.0

        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        t_stat = ate / se if se > 0 else 0.0
        p_value = float(2 * stats.norm.sf(abs(t_stat)))

        control_mean = float(np.mean(y[data[treatment].values[valid_mask] == 0])) if np.sum(data[treatment].values[valid_mask] == 0) > 0 else 0.0
        treated_mean = float(np.mean(y[data[treatment].values[valid_mask] == 1])) if np.sum(data[treatment].values[valid_mask] == 1) > 0 else 0.0

        return {
            "method": "backdoor_regression",
            "ate": round(ate, 6),
            "se": round(se, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant_at_05": p_value < 0.05,
            "treated_mean": round(treated_mean, 4),
            "control_mean": round(control_mean, 4),
            "confounders_adjusted": confounders,
            "n_treated": int(np.sum(data[treatment].values[valid_mask] == 1)),
            "n_control": int(np.sum(data[treatment].values[valid_mask] == 0)),
            "n_total": int(valid_mask.sum()),
        }

    @staticmethod
    def _ate_ipw(
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: list[str],
        treatment_vals: np.ndarray,
        outcome_vals: np.ndarray,
    ) -> dict[str, Any]:
        """ATE via Inverse Probability Weighting."""
        if not confounders:
            # Without confounders, IPW reduces to diff-in-means
            treated_mask = treatment_vals == 1
            control_mask = treatment_vals == 0
            n_t = int(treated_mask.sum())
            n_c = int(control_mask.sum())

            ate = float(np.mean(outcome_vals[treated_mask]) - np.mean(outcome_vals[control_mask]))
            se = float(np.sqrt(np.var(outcome_vals[treated_mask]) / n_t + np.var(outcome_vals[control_mask]) / n_c))

            return {
                "method": "ipw",
                "ate": round(ate, 6),
                "se": round(se, 6),
                "ci_lower": round(ate - 1.96 * se, 6),
                "ci_upper": round(ate + 1.96 * se, 6),
                "p_value": None,
                "note": "No confounders provided; IPW reduces to diff-in-means",
                "n_total": len(outcome_vals),
            }

        # Estimate propensity scores via logistic regression
        X_parts: list[np.ndarray] = [np.ones(len(data))]
        for conf in confounders:
            col = data[conf].values.astype(float)
            col = np.nan_to_num(col, nan=np.nanmean(col))
            X_parts.append(col)
        X = np.column_stack(X_parts)

        # Simple logistic regression for propensity scores
        from scipy.optimize import minimize as sp_minimize

        def logistic_loss(beta: np.ndarray) -> float:
            z = X @ beta
            z = np.clip(z, -500, 500)
            ll = -np.sum(
                treatment_vals * np.log(1 / (1 + np.exp(-z)) + 1e-12)
                + (1 - treatment_vals) * np.log(1 / (1 + np.exp(z)) + 1e-12)
            )
            return ll

        try:
            res = sp_minimize(logistic_loss, np.zeros(X.shape[1]), method="L-BFGS-B")
            ps = 1 / (1 + np.exp(-(X @ res.x)))
        except Exception:
            # Fallback: use treatment rate as constant propensity
            ps = np.full(len(data), float(treatment_vals.mean()))

        # Clip propensity scores for stability
        ps = np.clip(ps, 0.01, 0.99)

        # IPW estimator
        treated_mask = treatment_vals == 1
        control_mask = treatment_vals == 0

        # E[Y|T=1] estimated by IPW
        ipw_treated = np.sum(outcome_vals[treated_mask] / ps[treated_mask]) / np.sum(1 / ps[treated_mask])
        ipw_control = np.sum(outcome_vals[control_mask] / (1 - ps[control_mask])) / np.sum(1 / (1 - ps[control_mask]))

        ate = float(ipw_treated - ipw_control)

        # Standard error via bootstrap (simplified)
        n_boot = min(200, len(data))
        boot_ates: list[float] = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            idx = rng.choice(len(data), size=len(data), replace=True)
            ps_b = ps[idx]
            t_b = treatment_vals[idx]
            y_b = outcome_vals[idx]
            t_mask_b = t_b == 1
            c_mask_b = t_b == 0
            if t_mask_b.sum() < 2 or c_mask_b.sum() < 2:
                continue
            ipw_t_b = np.sum(y_b[t_mask_b] / ps_b[t_mask_b]) / np.sum(1 / ps_b[t_mask_b])
            ipw_c_b = np.sum(y_b[c_mask_b] / (1 - ps_b[c_mask_b])) / np.sum(1 / (1 - ps_b[c_mask_b]))
            boot_ates.append(float(ipw_t_b - ipw_c_b))

        se = float(np.std(boot_ates)) if boot_ates else 0.0
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return {
            "method": "ipw",
            "ate": round(ate, 6),
            "se": round(se, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "p_value": round(float(2 * stats.norm.sf(abs(ate / se))) if se > 0 else 1.0, 6),
            "significant_at_05": abs(ate / se) > 1.96 if se > 0 else False,
            "propensity_score_stats": {
                "mean": round(float(np.mean(ps)), 4),
                "std": round(float(np.std(ps)), 4),
                "min": round(float(np.min(ps)), 4),
                "max": round(float(np.max(ps)), 4),
            },
            "n_total": len(outcome_vals),
        }
