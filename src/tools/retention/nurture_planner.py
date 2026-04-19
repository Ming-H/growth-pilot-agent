"""New-user nurture / onboarding planner.

Generates day-1/7/14/30 action plans for new users based on behavioural
data and cohort retention curves, and evaluates nurture progress against
a baseline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Default nurture action templates
# -----------------------------------------------------------------------

_DEFAULT_PLAN: dict[str, list[dict[str, str]]] = {
    "day1": [
        {"action": "welcome_push", "channel": "push", "message": "欢迎使用！首单立减优惠"},
        {"action": "feature_tour", "channel": "in_app", "message": "3步完成你的第一次货运"},
    ],
    "day7": [
        {"action": "second_ride_incentive", "channel": "push", "message": "再来一单，运费再减5元"},
        {"action": "referral_invite", "channel": "in_app", "message": "邀请好友，各得10元券"},
    ],
    "day14": [
        {"action": "loyalty_intro", "channel": "email", "message": "累计3单解锁会员专属折扣"},
        {"action": "use_case_suggestion", "channel": "push", "message": "大件搬家？试试货运服务"},
    ],
    "day30": [
        {"action": "milestone_reward", "channel": "push", "message": "恭喜！解锁长期用户优惠"},
        {"action": "feedback_survey", "channel": "in_app", "message": "告诉我们你的体验"},
    ],
}

_DEFAULT_RETENTION_BASELINE: dict[str, float] = {
    "day1": 1.0,
    "day3": 0.65,
    "day7": 0.45,
    "day14": 0.30,
    "day30": 0.20,
}


class NurturePlanner:
    """Plan and evaluate new-user onboarding nurture campaigns.

    The planner generates a structured day-1/7/14/30 action plan and can
    compare an actual cohort's retention against a baseline curve.
    """

    def __init__(
        self,
        plan_template: dict[str, list[dict[str, str]]] | None = None,
        retention_baseline: dict[str, float] | None = None,
    ) -> None:
        self.plan_template = plan_template or _DEFAULT_PLAN
        self.retention_baseline = retention_baseline or _DEFAULT_RETENTION_BASELINE

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def generate_nurture_plan(
        self,
        new_user_data: pd.DataFrame | dict[str, Any] | None = None,
        retention_curves: dict[str, list[float]] | pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Generate a personalised nurture plan for new users.

        Parameters
        ----------
        new_user_data : optional
            User profile/segment information. Used to personalise plan
            content. If ``None``, returns the default template plan.
        retention_curves : optional
            Historical retention curves (day 1..30). Used to identify
            drop-off windows and intensify actions accordingly.

        Returns
        -------
        dict
            Keys: ``day1``, ``day7``, ``day14``, ``day30``, each containing
            a list of action dicts. Also includes ``"meta"`` with planning
            rationale.
        """
        plan: dict[str, Any] = {}
        for phase, actions in self.plan_template.items():
            plan[phase] = [dict(a) for a in actions]  # deep copy

        meta: dict[str, Any] = {"personalisation": "default"}

        # If retention curves are available, identify risk windows and
        # add extra actions at drop-off points.
        dropoff_phases = self._identify_dropoff_phases(retention_curves)
        if dropoff_phases:
            meta["dropoff_phases"] = dropoff_phases
            for phase in dropoff_phases:
                if phase in plan:
                    plan[phase].append(
                        {
                            "action": "dropoff_rescue",
                            "channel": "sms",
                            "message": f"我们在{phase}发现用户容易流失，专属优惠等你来",
                        }
                    )

        # Personalise based on user data
        if new_user_data is not None:
            plan, meta = self._personalise_plan(plan, meta, new_user_data)

        plan["meta"] = meta
        return plan

    # ------------------------------------------------------------------
    # Progress evaluation
    # ------------------------------------------------------------------

    def evaluate_nurture_progress(
        self,
        cohort_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Evaluate a cohort's actual retention against the baseline.

        Parameters
        ----------
        cohort_data : pd.DataFrame
            Must contain columns ``days_since_signup`` (int) and
            ``is_active`` (bool, where True = user still active).

        Returns
        -------
        dict
            Keys:
                ``actual_retention``: {day: rate},
                ``baseline_retention``: {day: rate},
                ``deviation``: {day: actual - baseline},
                ``overall_health``: "on_track" | "at_risk" | "underperforming",
                ``recommendations``: list of str
        """
        if cohort_data.empty:
            return {"overall_health": "no_data", "recommendations": ["Insufficient cohort data"]}

        df = cohort_data.copy()
        for col, default in [("days_since_signup", 0), ("is_active", False)]:
            if col not in df.columns:
                df[col] = default

        df["is_active"] = df["is_active"].astype(bool)
        df["days_since_signup"] = df["days_since_signup"].astype(int)

        # Compute actual retention at standard milestones
        milestones = [1, 3, 7, 14, 30]
        actual: dict[str, float] = {}
        for m in milestones:
            cohort_m = df[df["days_since_signup"] >= m]
            if len(cohort_m) == 0:
                continue
            active_at_m = cohort_m[cohort_m["is_active"]]
            # Users who signed up at least m days ago and are still active
            retained = len(active_at_m[active_at_m["days_since_signup"] >= m])
            total = len(cohort_m)
            rate = retained / total if total > 0 else 0.0
            actual[f"day{m}"] = round(rate, 4)

        # Compute deviations from baseline
        deviation: dict[str, float] = {}
        for day_key, actual_rate in actual.items():
            baseline_rate = self.retention_baseline.get(day_key, 0.0)
            deviation[day_key] = round(actual_rate - baseline_rate, 4)

        # Overall health assessment
        deviations = list(deviation.values())
        avg_dev = float(np.mean(deviations)) if deviations else 0.0
        if avg_dev >= -0.03:
            health = "on_track"
        elif avg_dev >= -0.10:
            health = "at_risk"
        else:
            health = "underperforming"

        # Recommendations
        recs = self._generate_recommendations(actual, deviation, health)

        return {
            "actual_retention": actual,
            "baseline_retention": dict(self.retention_baseline),
            "deviation": deviation,
            "overall_health": health,
            "recommendations": recs,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _identify_dropoff_phases(
        self,
        retention_curves: dict[str, list[float]] | pd.DataFrame | None,
    ) -> list[str]:
        """Identify phases with steep retention drops (>15% relative drop)."""
        if retention_curves is None:
            return []

        # Extract curve values
        if isinstance(retention_curves, pd.DataFrame):
            if "retention_rate" in retention_curves.columns:
                curve = retention_curves["retention_rate"].tolist()
            else:
                return []
        else:
            # dict of {phase: [values]}
            curve = []
            for phase in ("day1", "day7", "day14", "day30"):
                vals = retention_curves.get(phase)
                if vals:
                    curve.append(float(vals[-1]) if isinstance(vals, list) else float(vals))

        if len(curve) < 2:
            return []

        # Check for steep drops between consecutive points
        phases = ["day1", "day7", "day14", "day30"]
        dropoff: list[str] = []
        for i in range(1, min(len(curve), len(phases))):
            if curve[i - 1] > 0:
                relative_drop = (curve[i - 1] - curve[i]) / curve[i - 1]
                if relative_drop > 0.15:
                    dropoff.append(phases[i])

        return dropoff

    @staticmethod
    def _personalise_plan(
        plan: dict[str, Any],
        meta: dict[str, Any],
        new_user_data: pd.DataFrame | dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Adjust plan based on user data (city tier, etc.)."""
        if isinstance(new_user_data, dict):
            user_dict = new_user_data
        elif isinstance(new_user_data, pd.DataFrame):
            user_dict = new_user_data.iloc[0].to_dict() if len(new_user_data) > 0 else {}
        else:
            return plan, meta

        meta["personalisation"] = "personalised"

        city_tier = user_dict.get("city_tier", 3)
        if city_tier and int(city_tier) <= 2:
            # Tier-1/2 cities: emphasise speed and convenience
            plan["day1"].append(
                {"action": "premium_trial", "channel": "push", "message": "一线城市极速达，首单免费升级"}
            )
            meta["city_tier_personalised"] = True

        has_freight = user_dict.get("has_freight_search", False)
        if has_freight:
            plan["day1"].insert(0, {
                "action": "freight_onboarding",
                "channel": "in_app",
                "message": "你已搜索过货运，立即享受专属优惠",
            })
            meta["freight_interest_personalised"] = True

        return plan, meta

    @staticmethod
    def _generate_recommendations(
        actual: dict[str, float],
        deviation: dict[str, float],
        health: str,
    ) -> list[str]:
        """Generate actionable recommendations based on retention analysis."""
        recs: list[str] = []

        if health == "underperforming":
            recs.append("Overall retention is significantly below baseline. Consider revisiting the entire onboarding flow.")

        # Check specific phases
        for phase, dev in deviation.items():
            if dev < -0.10:
                recs.append(
                    f"{phase} retention is {abs(dev)*100:.1f}pp below baseline. "
                    "Intensify engagement actions at this stage."
                )

        if actual.get("day1", 1.0) < 0.95:
            recs.append("Day-1 retention drop detected. Review welcome experience and first-use friction.")

        if actual.get("day7", 0) < 0.30:
            recs.append("Week-1 retention is low. Strengthen second-ride incentive and push notification strategy.")

        if actual.get("day30", 0) < 0.15:
            recs.append("30-day retention critically low. Introduce loyalty programme or milestone rewards.")

        if not recs:
            recs.append("Retention is on track. Continue monitoring and A/B test incremental improvements.")

        return recs
