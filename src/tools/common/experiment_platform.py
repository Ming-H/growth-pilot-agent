"""Experiment platform: design, analyze, and prioritize A/B tests.

Provides statistical methods for experiment design (power analysis),
analysis (t-test, chi-square, sequential SPRT, Bayesian), and
prioritization (ICE scoring).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats


class ExperimentPlatform:
    """Design, analyze, and monitor A/B/n experiments.

    This is the central experimentation toolkit shared by all agents
    in the growth-pilot system.
    """

    # ------------------------------------------------------------------
    # Experiment design
    # ------------------------------------------------------------------

    def design_experiment(
        self,
        hypothesis: str,
        metric: str,
        guardrail_metrics: list[str],
        expected_lift: float,
        baseline_rate: float,
        n_variants: int = 2,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> dict[str, Any]:
        """Design an experiment with sample size calculation.

        Uses power analysis to determine minimum sample size per variant,
        then estimates experiment duration based on daily traffic.

        Args:
            hypothesis: Description of the hypothesis being tested.
            metric: Primary metric name (e.g. 'conversion_rate').
            guardrail_metrics: Metrics that must not degrade (e.g. ['churn_rate']).
            expected_lift: Expected relative lift (e.g. 0.05 for 5% lift).
            baseline_rate: Current baseline conversion rate (0.0 ~ 1.0).
            n_variants: Number of variants including control (default 2).
            alpha: Significance level (default 0.05).
            power: Statistical power (default 0.8).

        Returns:
            Dict with experiment design parameters:
                - 'hypothesis', 'metric', 'guardrail_metrics'
                - 'baseline_rate', 'mde' (minimum detectable effect)
                - 'expected_lift', 'target_rate'
                - 'sample_size_per_variant'
                - 'total_sample_size'
                - 'n_variants', 'alpha', 'power'
                - 'estimated_duration_days' (assuming 10k daily traffic)
        """
        if baseline_rate <= 0 or baseline_rate >= 1:
            return {"error": "baseline_rate must be between 0 and 1 (exclusive)"}
        if expected_lift <= 0:
            return {"error": "expected_lift must be positive"}

        target_rate = baseline_rate * (1 + expected_lift)
        if target_rate >= 1:
            return {"error": "target_rate exceeds 1; reduce expected_lift"}

        # Minimum Detectable Effect in absolute terms
        mde_absolute = abs(target_rate - baseline_rate)

        # Sample size per variant (two-proportion z-test power analysis)
        # n = (Z_alpha/2 + Z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p2-p1)^2
        z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
        z_beta = sp_stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = target_rate

        numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
        denominator = (p2 - p1) ** 2

        if denominator == 0:
            return {"error": "Cannot compute sample size: target equals baseline"}

        sample_size_per_variant = int(np.ceil(numerator / denominator))
        total_sample_size = sample_size_per_variant * n_variants

        # Assume 10k daily traffic for duration estimation
        daily_traffic = 10000
        estimated_duration_days = int(np.ceil(total_sample_size / daily_traffic))

        return {
            "hypothesis": hypothesis,
            "metric": metric,
            "guardrail_metrics": guardrail_metrics,
            "baseline_rate": baseline_rate,
            "mde": round(mde_absolute, 6),
            "expected_lift": expected_lift,
            "target_rate": round(target_rate, 6),
            "sample_size_per_variant": sample_size_per_variant,
            "total_sample_size": total_sample_size,
            "n_variants": n_variants,
            "alpha": alpha,
            "power": power,
            "estimated_duration_days": estimated_duration_days,
            "daily_traffic_assumed": daily_traffic,
        }

    # ------------------------------------------------------------------
    # Experiment analysis
    # ------------------------------------------------------------------

    def analyze_experiment(
        self,
        experiment_data: list[dict[str, Any]],
        metric: str,
        method: str = "t_test",
    ) -> dict[str, Any]:
        """Analyze experiment results with statistical testing.

        Args:
            experiment_data: List of dicts, each with keys:
                - 'group': str (e.g. 'control', 'treatment')
                - metric: the metric value for this observation
            metric: Column name to analyze.
            method: Statistical method: 't_test', 'chi_square', 'sequential', 'bayesian'.

        Returns:
            Dict with:
                - 'method': the method used
                - 'groups': per-group summary statistics
                - 'lift': relative lift of treatment vs control
                - 'confidence_interval': 95% CI for the difference
                - 'p_value': statistical p-value
                - 'significant': whether result is significant at alpha=0.05
                - 'srm_check': sample ratio mismatch test result
        """
        if not experiment_data:
            return {"error": "No experiment data provided"}

        # Group data
        groups: dict[str, list[float]] = {}
        for row in experiment_data:
            grp = str(row.get("group", ""))
            val = row.get(metric)
            if val is None:
                continue
            groups.setdefault(grp, []).append(float(val))

        if len(groups) < 2:
            return {"error": "Need at least 2 groups for analysis"}

        # Per-group statistics
        group_stats: dict[str, dict[str, Any]] = {}
        for grp_name, values in groups.items():
            arr = np.array(values)
            group_stats[grp_name] = {
                "count": len(arr),
                "mean": round(float(np.mean(arr)), 6),
                "std": round(float(np.std(arr, ddof=1)), 6),
                "sum": round(float(np.sum(arr)), 4),
            }

        # SRM check (Sample Ratio Mismatch)
        srm_check = self._srm_check(groups)

        # Identify control and treatment
        group_names = list(groups.keys())
        control_name = group_names[0] if "control" not in group_names else "control"
        treatment_names = [g for g in group_names if g != control_name]

        control_vals = np.array(groups[control_name])
        control_mean = float(np.mean(control_vals))

        # If there's only one treatment group, compute pairwise results
        results_per_comparison: list[dict[str, Any]] = []
        for treat_name in treatment_names:
            treat_vals = np.array(groups[treat_name])
            treat_mean = float(np.mean(treat_vals))

            # Lift
            lift = (treat_mean - control_mean) / control_mean if control_mean != 0 else 0.0

            # Choose method
            if method == "t_test":
                test_result = self._t_test(control_vals, treat_vals)
            elif method == "chi_square":
                test_result = self._chi_square_test(groups[control_name], groups[treat_name])
            elif method == "sequential":
                test_result = self._sprt_test(groups[control_name], groups[treat_name])
            elif method == "bayesian":
                test_result = self._bayesian_test(control_vals, treat_vals)
            else:
                return {"error": f"Unknown method '{method}'. Use t_test, chi_square, sequential, or bayesian"}

            comparison = {
                "control": control_name,
                "treatment": treat_name,
                "control_mean": round(control_mean, 6),
                "treatment_mean": round(treat_mean, 6),
                "lift": round(lift, 4),
                **test_result,
            }
            results_per_comparison.append(comparison)

        # For single comparison (2 groups), flatten results
        if len(treatment_names) == 1:
            primary_result = results_per_comparison[0]
        else:
            primary_result = results_per_comparison

        return {
            "method": method,
            "metric": metric,
            "groups": group_stats,
            "lift": results_per_comparison[0]["lift"] if results_per_comparison else None,
            "confidence_interval": results_per_comparison[0].get("confidence_interval") if results_per_comparison else None,
            "p_value": results_per_comparison[0].get("p_value") if results_per_comparison else None,
            "significant": results_per_comparison[0].get("significant", False) if results_per_comparison else False,
            "srm_check": srm_check,
            "comparisons": results_per_comparison,
        }

    # ------------------------------------------------------------------
    # Sequential testing (SPRT)
    # ------------------------------------------------------------------

    def sequential_test(
        self,
        daily_data: list[dict[str, Any]],
        metric: str,
        method: str = "sprt",
        alpha_spend: float = 0.05,
    ) -> dict[str, Any]:
        """Run sequential monitoring on daily experiment data with optional early stopping.

        Applies the Sequential Probability Ratio Test (SPRT) or alpha-spending
        approach to check if results are significant at each interim look.

        Args:
            daily_data: List of dicts, each with keys:
                - 'day': int (day index starting from 1)
                - 'group': str ('control' or 'treatment')
                - metric: the metric value
            metric: Column name to analyze.
            method: 'sprt' for SPRT or 'alpha_spend' for O'Brien-Fleming spending.
            alpha_spend: Total alpha to spend (default 0.05).

        Returns:
            Dict with:
                - 'daily_results': list of daily check results
                - 'stopped_early': whether early stopping was triggered
                - 'stop_day': day index where stopped (None if never)
                - 'final_verdict': 'significant' | 'not_significant' | 'continue'
                - 'cumulative_lift': final cumulative lift
        """
        if not daily_data:
            return {
                "daily_results": [],
                "stopped_early": False,
                "stop_day": None,
                "final_verdict": "not_significant",
                "cumulative_lift": 0.0,
            }

        # Group by day
        day_groups: dict[int, dict[str, list[float]]] = {}
        for row in daily_data:
            day = row.get("day", 1)
            grp = str(row.get("group", ""))
            val = row.get(metric)
            if val is None:
                continue
            day_groups.setdefault(day, {}).setdefault(grp, []).append(float(val))

        sorted_days = sorted(day_groups.keys())

        # Cumulative accumulation
        cumulative_control: list[float] = []
        cumulative_treatment: list[float] = []
        daily_results: list[dict[str, Any]] = []
        stopped_early = False
        stop_day: int | None = None

        for day in sorted_days:
            day_data = day_groups[day]
            cumulative_control.extend(day_data.get("control", []))
            cumulative_treatment.extend(day_data.get("treatment", []))

            if len(cumulative_control) < 10 or len(cumulative_treatment) < 10:
                daily_results.append({
                    "day": day,
                    "status": "insufficient_data",
                    "n_control": len(cumulative_control),
                    "n_treatment": len(cumulative_treatment),
                })
                continue

            ctrl_arr = np.array(cumulative_control)
            treat_arr = np.array(cumulative_treatment)

            ctrl_mean = float(np.mean(ctrl_arr))
            treat_mean = float(np.mean(treat_arr))
            lift = (treat_mean - ctrl_mean) / ctrl_mean if ctrl_mean != 0 else 0.0

            if method == "sprt":
                # SPRT: compute log-likelihood ratio
                # Using normal approximation
                ctrl_std = float(np.std(ctrl_arr, ddof=1))
                treat_std = float(np.std(treat_arr, ddof=1))
                pooled_std = np.sqrt(ctrl_std**2 + treat_std**2)

                if pooled_std == 0:
                    sprt_stat = 0.0
                else:
                    sprt_stat = (treat_mean - ctrl_mean) / pooled_std * np.sqrt(
                        min(len(ctrl_arr), len(treat_arr))
                    )

                # SPRT boundaries
                upper_bound = np.log((1 - alpha_spend / 2) / (alpha_spend / 2))
                lower_bound = -upper_bound

                if sprt_stat > upper_bound:
                    status = "significant_positive"
                elif sprt_stat < lower_bound:
                    status = "significant_negative"
                else:
                    status = "continue"

                daily_results.append({
                    "day": day,
                    "status": status,
                    "sprt_statistic": round(sprt_stat, 4),
                    "upper_bound": round(float(upper_bound), 4),
                    "lower_bound": round(float(lower_bound), 4),
                    "lift": round(lift, 4),
                    "n_control": len(cumulative_control),
                    "n_treatment": len(cumulative_treatment),
                })

                if status.startswith("significant"):
                    stopped_early = True
                    stop_day = day
                    break

            elif method == "alpha_spend":
                # O'Brien-Fleming alpha spending function
                look_idx = len([d for d in daily_results if d.get("status") != "insufficient_data"]) + 1
                info_fraction = min(look_idx / max(len(sorted_days), 1), 1.0)
                # OBF spending: 2 - 2*Phi(z_alpha/4 / sqrt(t))
                z_val = sp_stats.norm.ppf(1 - alpha_spend / 4) / np.sqrt(info_fraction) if info_fraction > 0 else float("inf")
                alpha_spent = 2 * (1 - sp_stats.norm.cdf(z_val))

                # Z-test at this look
                ctrl_std = float(np.std(ctrl_arr, ddof=1))
                treat_std = float(np.std(treat_arr, ddof=1))
                se = np.sqrt(ctrl_std**2 / len(ctrl_arr) + treat_std**2 / len(treat_arr))

                if se == 0:
                    z_score = 0.0
                    p_val = 1.0
                else:
                    z_score = (treat_mean - ctrl_mean) / se
                    p_val = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))

                significant = p_val < alpha_spent

                daily_results.append({
                    "day": day,
                    "status": "significant" if significant else "continue",
                    "z_score": round(float(z_score), 4),
                    "p_value": round(float(p_val), 6),
                    "alpha_spent": round(float(alpha_spent), 6),
                    "lift": round(lift, 4),
                    "n_control": len(cumulative_control),
                    "n_treatment": len(cumulative_treatment),
                })

                if significant:
                    stopped_early = True
                    stop_day = day
                    break
            else:
                return {"error": f"Unknown sequential method '{method}'. Use 'sprt' or 'alpha_spend'"}

        # Final verdict
        if stopped_early:
            final_verdict = "significant"
        elif daily_results and daily_results[-1].get("status") == "continue":
            final_verdict = "not_significant"
        else:
            final_verdict = "continue"

        # Cumulative lift
        final_lift = 0.0
        if cumulative_control and cumulative_treatment:
            cm = float(np.mean(cumulative_control))
            tm = float(np.mean(cumulative_treatment))
            final_lift = round((tm - cm) / cm, 4) if cm != 0 else 0.0

        return {
            "daily_results": daily_results,
            "stopped_early": stopped_early,
            "stop_day": stop_day,
            "final_verdict": final_verdict,
            "cumulative_lift": final_lift,
        }

    # ------------------------------------------------------------------
    # Experiment prioritization
    # ------------------------------------------------------------------

    def prioritize_experiments(
        self,
        backlog: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prioritize experiment backlog by ICE score.

        ICE = Impact * Confidence * Ease, each scored 1-10.

        Args:
            backlog: List of dicts, each with keys:
                - 'name': str
                - 'description': str (optional)
                - 'impact': float (1-10)
                - 'confidence': float (1-10)
                - 'ease': float (1-10)

        Returns:
            List sorted by ICE score descending, with 'ice_score' added to each.
        """
        if not backlog:
            return []

        scored: list[dict[str, Any]] = []
        for item in backlog:
            impact = float(item.get("impact", 0))
            confidence = float(item.get("confidence", 0))
            ease = float(item.get("ease", 0))

            # Clamp to 1-10 range
            impact = max(1, min(10, impact))
            confidence = max(1, min(10, confidence))
            ease = max(1, min(10, ease))

            ice_score = round(impact * confidence * ease, 2)

            scored.append({
                **item,
                "impact": impact,
                "confidence": confidence,
                "ease": ease,
                "ice_score": ice_score,
            })

        scored.sort(key=lambda x: x["ice_score"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Private statistical methods
    # ------------------------------------------------------------------

    @staticmethod
    def _t_test(control: np.ndarray, treatment: np.ndarray) -> dict[str, Any]:
        """Welch's t-test for comparing two group means."""
        t_stat, p_value = sp_stats.ttest_ind(control, treatment, equal_var=False)

        # Confidence interval for difference in means
        n1, n2 = len(control), len(treatment)
        mean_diff = float(np.mean(treatment)) - float(np.mean(control))
        se = np.sqrt(np.var(control, ddof=1) / n1 + np.var(treatment, ddof=1) / n2)
        ci_lower = mean_diff - 1.96 * se
        ci_upper = mean_diff + 1.96 * se

        return {
            "p_value": round(float(p_value), 6),
            "t_statistic": round(float(t_stat), 4),
            "confidence_interval": (round(float(ci_lower), 6), round(float(ci_upper), 6)),
            "significant": p_value < 0.05,
        }

    @staticmethod
    def _chi_square_test(
        control: list[float],
        treatment: list[float],
    ) -> dict[str, Any]:
        """Chi-square test for comparing binary outcome rates.

        Treats values as binary (0/1) and tests independence.
        """
        ctrl_arr = np.array(control)
        treat_arr = np.array(treatment)

        # Build 2x2 contingency table
        ctrl_success = int(np.sum(ctrl_arr))
        ctrl_failure = len(ctrl_arr) - ctrl_success
        treat_success = int(np.sum(treat_arr))
        treat_failure = len(treat_arr) - treat_success

        table = np.array([
            [ctrl_success, ctrl_failure],
            [treat_success, treat_failure],
        ])

        chi2, p_value, dof, expected = sp_stats.chi2_contingency(table, correction=True)

        return {
            "p_value": round(float(p_value), 6),
            "chi2_statistic": round(float(chi2), 4),
            "dof": int(dof),
            "confidence_interval": None,
            "significant": p_value < 0.05,
        }

    @staticmethod
    def _sprt_test(
        control: list[float],
        treatment: list[float],
    ) -> dict[str, Any]:
        """SPRT-based test (single look) for two groups."""
        ctrl_arr = np.array(control)
        treat_arr = np.array(treatment)

        ctrl_mean = float(np.mean(ctrl_arr))
        treat_mean = float(np.mean(treat_arr))

        ctrl_std = float(np.std(ctrl_arr, ddof=1)) if len(ctrl_arr) > 1 else 1.0
        treat_std = float(np.std(treat_arr, ddof=1)) if len(treat_arr) > 1 else 1.0

        pooled_se = np.sqrt(ctrl_std**2 / len(ctrl_arr) + treat_std**2 / len(treat_arr))
        if pooled_se == 0:
            return {
                "p_value": None,
                "sprt_statistic": None,
                "confidence_interval": None,
                "significant": False,
            }

        z = (treat_mean - ctrl_mean) / pooled_se
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

        ci_lower = (treat_mean - ctrl_mean) - 1.96 * pooled_se
        ci_upper = (treat_mean - ctrl_mean) + 1.96 * pooled_se

        return {
            "p_value": round(float(p_value), 6),
            "sprt_statistic": round(float(z), 4),
            "confidence_interval": (round(float(ci_lower), 6), round(float(ci_upper), 6)),
            "significant": p_value < 0.05,
        }

    @staticmethod
    def _bayesian_test(
        control: np.ndarray,
        treatment: np.ndarray,
        n_simulations: int = 50000,
    ) -> dict[str, Any]:
        """Bayesian A/B test using conjugate Beta-Binomial model.

        Assumes binary outcomes (0/1). Uses Beta priors and posterior
        sampling to estimate P(treatment > control).
        """
        rng = np.random.default_rng(42)

        # Beta prior (weakly informative)
        alpha_prior, beta_prior = 1.0, 1.0

        ctrl_successes = int(np.sum(control))
        ctrl_trials = len(control)
        treat_successes = int(np.sum(treatment))
        treat_trials = len(treatment)

        # Posterior parameters
        ctrl_alpha = alpha_prior + ctrl_successes
        ctrl_beta = beta_prior + ctrl_trials - ctrl_successes
        treat_alpha = alpha_prior + treat_successes
        treat_beta = beta_prior + treat_trials - treat_successes

        # Sample from posteriors
        ctrl_samples = rng.beta(ctrl_alpha, ctrl_beta, size=n_simulations)
        treat_samples = rng.beta(treat_alpha, treat_beta, size=n_simulations)

        prob_treatment_better = float(np.mean(treat_samples > ctrl_samples))

        # Credible interval for the difference
        diff_samples = treat_samples - ctrl_samples
        ci_lower = float(np.percentile(diff_samples, 2.5))
        ci_upper = float(np.percentile(diff_samples, 97.5))

        # Expected lift
        ctrl_posterior_mean = ctrl_alpha / (ctrl_alpha + ctrl_beta)
        treat_posterior_mean = treat_alpha / (treat_alpha + treat_beta)
        lift = (treat_posterior_mean - ctrl_posterior_mean) / ctrl_posterior_mean if ctrl_posterior_mean > 0 else 0.0

        return {
            "p_value": None,  # Bayesian: no p-value
            "prob_treatment_better": round(prob_treatment_better, 4),
            "expected_lift": round(lift, 4),
            "credible_interval": (round(ci_lower, 6), round(ci_upper, 6)),
            "confidence_interval": (round(ci_lower, 6), round(ci_upper, 6)),
            "significant": prob_treatment_better > 0.95,
            "control_posterior_mean": round(ctrl_posterior_mean, 6),
            "treatment_posterior_mean": round(treat_posterior_mean, 6),
        }

    @staticmethod
    def _srm_check(groups: dict[str, list[float]]) -> dict[str, Any]:
        """Check for Sample Ratio Mismatch using chi-square test.

        Tests whether the observed group sizes match the expected 50/50 split.
        """
        observed = np.array([len(v) for v in groups.values()])
        total = observed.sum()
        expected = np.full_like(observed, total / len(observed), dtype=float)

        if total == 0:
            return {"srm_detected": False, "p_value": None, "note": "No data"}

        chi2 = float(np.sum((observed - expected) ** 2 / expected))
        dof = len(observed) - 1
        p_value = float(1 - sp_stats.chi2.cdf(chi2, dof))

        return {
            "srm_detected": p_value < 0.01,
            "p_value": round(p_value, 6),
            "chi2": round(chi2, 4),
            "observed_ratios": {k: round(len(v) / total, 4) for k, v in groups.items()},
        }
