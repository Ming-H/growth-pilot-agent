"""Visualization tools for growth metrics.

Generates matplotlib/seaborn charts for funnels, retention curves,
attribution comparisons, seasonal trends, and experiment results.
All plots use a consistent style and return the saved file path when
a save_path is provided.
"""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from src.tools.registry import ToolRegistry

# Consistent style
sns.set_theme(style="whitegrid", font="sans-serif", palette="muted")


@ToolRegistry.register("visualizer")
class Visualizer:
    """Generate charts for growth-pilot metrics."""

    # ------------------------------------------------------------------
    # Funnel
    # ------------------------------------------------------------------

    def plot_funnel(
        self,
        funnel_data: list[dict[str, Any]],
        save_path: str | None = None,
    ) -> str:
        """Plot a funnel chart from stage data.

        Args:
            funnel_data: List of dicts with 'stage' and 'count' keys,
                         ordered from top to bottom of the funnel.
            save_path: Optional file path to save the chart (PNG).

        Returns:
            The save_path if provided, else empty string.
        """
        if not funnel_data:
            return ""

        stages = [d.get("stage", "") for d in funnel_data]
        counts = [d.get("count", 0) for d in funnel_data]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Horizontal bar chart with decreasing widths for funnel effect
        max_count = max(counts) if counts else 1
        colors = sns.color_palette("Blues_d", len(stages))

        for i, (stage, count) in enumerate(zip(stages, counts)):
            width = count / max_count
            left = (1 - width) / 2
            ax.barh(len(stages) - 1 - i, width, left=left, height=0.6,
                    color=colors[i], edgecolor="white", linewidth=1.5)
            ax.text(0.5, len(stages) - 1 - i,
                    f"{stage}\n{count:,} ({count/max_count*100:.1f}%)",
                    ha="center", va="center", fontsize=11, fontweight="bold")

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(stages) - 0.5)
        ax.axis("off")
        ax.set_title("Conversion Funnel", fontsize=14, fontweight="bold", pad=20)

        # Add conversion rate annotations between stages
        for i in range(len(counts) - 1):
            if counts[i] > 0:
                rate = counts[i + 1] / counts[i] * 100
                mid_y = len(stages) - 1 - i - 0.5
                ax.text(0.92, mid_y, f"{rate:.1f}%", ha="center", va="center",
                        fontsize=9, color="#666666", style="italic")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path or ""

    # ------------------------------------------------------------------
    # Retention curve
    # ------------------------------------------------------------------

    def plot_retention_curve(
        self,
        cohort_data: list[dict[str, Any]],
        save_path: str | None = None,
    ) -> str:
        """Plot retention curves for multiple cohorts.

        Args:
            cohort_data: List of dicts with 'cohort', 'period', 'retention_rate'.
            save_path: Optional file path to save the chart (PNG).

        Returns:
            The save_path if provided, else empty string.
        """
        if not cohort_data:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by cohort
        cohorts: dict[str, list[dict[str, Any]]] = {}
        for d in cohort_data:
            cohorts.setdefault(d.get("cohort", ""), []).append(d)

        colors = sns.color_palette("husl", len(cohorts))
        for i, (cohort_name, entries) in enumerate(sorted(cohorts.items())):
            entries.sort(key=lambda x: x.get("period", 0))
            periods = [e.get("period", 0) for e in entries]
            rates = [e.get("retention_rate", 0) * 100 for e in entries]
            ax.plot(periods, rates, marker="o", markersize=4, linewidth=2,
                    label=cohort_name, color=colors[i])

        ax.set_xlabel("Period (Months)", fontsize=12)
        ax.set_ylabel("Retention Rate (%)", fontsize=12)
        ax.set_title("Cohort Retention Curves", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter()
                                     if max(ax.get_ylim()) <= 100
                                     else mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path or ""

    # ------------------------------------------------------------------
    # Attribution comparison
    # ------------------------------------------------------------------

    def plot_attribution_comparison(
        self,
        attribution_result: dict[str, dict[str, float]],
        save_path: str | None = None,
    ) -> str:
        """Plot grouped bar chart comparing attribution across models.

        Args:
            attribution_result: Dict of {model_name: {channel: attribution_value}},
                                e.g. {"first_touch": {"搜索": 0.3, "推送": 0.2}, ...}
            save_path: Optional file path to save the chart (PNG).

        Returns:
            The save_path if provided, else empty string.
        """
        if not attribution_result:
            return ""

        models = list(attribution_result.keys())
        # Collect all channels across all models
        all_channels: list[str] = sorted({
            ch for model in attribution_result.values() for ch in model.keys()
        })

        if not all_channels:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(all_channels))
        width = 0.8 / len(models)

        colors = sns.color_palette("Set2", len(models))
        for i, model in enumerate(models):
            values = [attribution_result[model].get(ch, 0.0) for ch in all_channels]
            bars = ax.bar(x + i * width, values, width, label=model, color=colors[i])
            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Channel", fontsize=12)
        ax.set_ylabel("Attribution Weight", fontsize=12)
        ax.set_title("Attribution Model Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(all_channels, rotation=30, ha="right")
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path or ""

    # ------------------------------------------------------------------
    # Seasonal trend
    # ------------------------------------------------------------------

    def plot_seasonal_trend(
        self,
        daily_metrics: list[dict[str, Any]],
        save_path: str | None = None,
    ) -> str:
        """Plot seasonal trend with rolling average.

        Args:
            daily_metrics: List of dicts with 'date' and at least one of:
                           'orders', 'revenue', 'new_users'.
            save_path: Optional file path to save the chart (PNG).

        Returns:
            The save_path if provided, else empty string.
        """
        if not daily_metrics:
            return ""

        dates = [d.get("date", "") for d in daily_metrics]
        orders = [d.get("orders", 0) for d in daily_metrics]

        fig, ax1 = plt.subplots(figsize=(14, 5))

        # Daily orders
        ax1.fill_between(range(len(dates)), orders, alpha=0.2, color="steelblue")
        ax1.plot(range(len(dates)), orders, linewidth=0.8, color="steelblue", alpha=0.5, label="Daily Orders")

        # 7-day rolling average
        if len(orders) >= 7:
            rolling = np.convolve(orders, np.ones(7) / 7, mode="valid")
            ax1.plot(range(3, 3 + len(rolling)), rolling,
                     linewidth=2.5, color="darkblue", label="7-Day Avg")

        # Revenue on secondary axis (if available)
        revenue = [d.get("revenue", None) for d in daily_metrics]
        if any(r is not None and r > 0 for r in revenue):
            ax2 = ax1.twinx()
            revenue_vals = [r if r is not None else 0 for r in revenue]
            ax2.plot(range(len(dates)), revenue_vals,
                     linewidth=0.8, color="coral", alpha=0.5, label="Daily Revenue")
            ax2.set_ylabel("Revenue (CNY)", fontsize=11, color="coral")
            ax2.tick_params(axis="y", labelcolor="coral")

            # Rolling avg for revenue
            if len(revenue_vals) >= 7:
                rev_rolling = np.convolve(revenue_vals, np.ones(7) / 7, mode="valid")
                ax2.plot(range(3, 3 + len(rev_rolling)), rev_rolling,
                         linewidth=2.5, color="firebrick", label="7-Day Rev Avg")

        # X-axis: show monthly labels
        step = max(1, len(dates) // 12)
        tick_positions = list(range(0, len(dates), step))
        tick_labels = [dates[i] for i in tick_positions]
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels, rotation=30, ha="right")

        ax1.set_xlabel("Date", fontsize=11)
        ax1.set_ylabel("Orders", fontsize=11, color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.set_title("Seasonal Trend - Orders & Revenue", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path or ""

    # ------------------------------------------------------------------
    # Experiment results
    # ------------------------------------------------------------------

    def plot_experiment_results(
        self,
        experiment_result: dict[str, Any],
        save_path: str | None = None,
    ) -> str:
        """Plot experiment results with confidence intervals.

        Args:
            experiment_result: Dict from ExperimentPlatform.analyze_experiment(), with:
                - 'groups': {group_name: {'mean', 'std', 'count'}}
                - 'lift': float
                - 'significant': bool
                - 'comparisons': list of comparison dicts with 'confidence_interval'
            save_path: Optional file path to save the chart (PNG).

        Returns:
            The save_path if provided, else empty string.
        """
        if not experiment_result:
            return ""

        groups = experiment_result.get("groups", {})
        if not groups:
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # --- Left panel: Group means with error bars ---
        ax1 = axes[0]
        group_names = list(groups.keys())
        means = [groups[g].get("mean", 0) for g in group_names]
        stds = [groups[g].get("std", 0) for g in group_names]
        counts = [groups[g].get("count", 1) for g in group_names]
        sems = [s / np.sqrt(n) if n > 0 else 0 for s, n in zip(stds, counts)]

        colors = ["#4C72B0" if i == 0 else "#DD8452" for i in range(len(group_names))]
        bars = ax1.bar(group_names, means, yerr=[1.96 * s for s in sems],
                       capsize=8, color=colors, edgecolor="white", linewidth=1.5)

        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{mean:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax1.set_title("Group Means (95% CI)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Metric Value", fontsize=11)

        # --- Right panel: Lift and significance ---
        ax2 = axes[1]
        lift = experiment_result.get("lift", 0)
        significant = experiment_result.get("significant", False)
        comparisons = experiment_result.get("comparisons", [])

        # Plot lift as a bar
        lift_color = "#55A868" if significant and lift > 0 else "#C44E52" if significant else "#BBBBBB"
        ax2.bar(["Lift"], [lift * 100], color=lift_color, edgecolor="white", linewidth=1.5)

        # CI for lift (from first comparison)
        if comparisons:
            ci = comparisons[0].get("confidence_interval")
            if ci and ci[0] is not None:
                ctrl_mean = comparisons[0].get("control_mean", 1)
                if ctrl_mean and ctrl_mean != 0:
                    ci_pct = [(v / ctrl_mean) * 100 for v in ci]
                    ax2.errorbar(["Lift"], [lift * 100],
                                 yerr=[[lift * 100 - ci_pct[0]], [ci_pct[1] - lift * 100]],
                                 fmt="none", ecolor="black", capsize=10, linewidth=2)

        significance_text = "Significant" if significant else "Not Significant"
        ax2.set_title(f"Lift: {lift*100:.2f}% ({significance_text})",
                      fontsize=12, fontweight="bold",
                      color="green" if significant and lift > 0 else "red" if significant else "gray")
        ax2.set_ylabel("Lift (%)", fontsize=11)
        ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

        # Add p-value annotation
        p_value = experiment_result.get("p_value")
        if p_value is not None:
            ax2.text(0.5, 0.02, f"p-value: {p_value:.4f}",
                     transform=ax2.transAxes, ha="center", fontsize=10, color="#666666")

        plt.suptitle("Experiment Results", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path or ""
