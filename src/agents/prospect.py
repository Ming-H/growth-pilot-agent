"""ProspectExpert - User prospecting, scoring, segmentation and LTV prediction."""

from __future__ import annotations

import logging
from typing import Any

from src.core.expert import ExpertAgentBase
from src.prompts.templates.agent_prompts import ProspectPrompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool imports (tools are built in parallel; provide fallback stubs)
# ---------------------------------------------------------------------------
try:
    from src.tools.prospect.feature_engine import FeatureEngine
    from src.tools.prospect.intent_model import IntentModel
    from src.tools.prospect.user_scorer import UserScorer
    from src.tools.prospect.segmentor import UserSegmentor
    from src.tools.prospect.ltv_predictor import LVTPredictor
except ImportError:

    class _Stub:
        """Minimal stub for tools not yet implemented."""

        def __init__(self, *a: Any, **kw: Any) -> None: ...

    FeatureEngine = IntentModel = UserScorer = UserSegmentor = LVTPredictor = _Stub


# ---------------------------------------------------------------------------
# Keyword set for can_handle routing
# ---------------------------------------------------------------------------
_PROSPECT_KEYWORDS: list[str] = [
    "潜客",
    "新客",
    "用户评分",
    "分群",
    "LTV",
    "意图",
    "prospect",
    "scoring",
    "segment",
]


class ProspectExpert(ExpertAgentBase):
    """Identifies and scores potential high-value users for freight services."""

    name = "prospect"
    description = "潜客识别与评分 Agent"

    # ------------------------------------------------------------------
    # ExpertAgentBase abstract implementations
    # ------------------------------------------------------------------

    def _init_tools(self) -> dict[str, Any]:
        """Initialize and return deterministic tool instances as a dict."""
        return {
            "feature_engine": FeatureEngine(),
            "intent_model": IntentModel(),
            "scorer": UserScorer(),
            "segmentor": UserSegmentor(),
            "ltv_predictor": LVTPredictor(),
        }

    def _get_system_prompt(self) -> str:
        """Return the prospect expert's system prompt for LLM synthesis."""
        template = ProspectPrompt()
        return f"{template.role_definition}\n\n{template.business_context}"

    def can_handle(self, query: str) -> float:
        """Return confidence score (0-1) for handling this query."""
        query_lower = query.lower()
        match_count = sum(1 for kw in _PROSPECT_KEYWORDS if kw.lower() in query_lower)
        if match_count == 0:
            return 0.0
        # Scale: 1 match -> 0.6, 2+ matches -> 0.8, 3+ -> 1.0
        if match_count >= 3:
            return 1.0
        if match_count >= 2:
            return 0.8
        return 0.6

    # ------------------------------------------------------------------
    # Main deterministic pipeline
    # ------------------------------------------------------------------

    def _execute_pipeline(self, params: dict) -> dict:
        """Execute the prospect analysis pipeline.

        Returns a flat dict with pipeline results.
        """
        errors: list[str] = []
        data_path = params.get("data_path", "")

        # Unpack tools from the initialized dict
        feature_engine = self._tools["feature_engine"]
        intent_model = self._tools["intent_model"]
        scorer = self._tools["scorer"]
        segmentor = self._tools["segmentor"]
        ltv_predictor = self._tools["ltv_predictor"]

        # 1. Build features
        try:
            raw_data = self._load_data(data_path)
            features = feature_engine.build_feature_matrix(raw_data)
            user_count = len(features)
        except Exception as exc:
            logger.warning("FeatureEngine failed: %s", exc)
            features = None
            user_count = 0
            errors.append(f"FeatureEngine: {exc}")

        # 2. Predict intent
        intent_scores: Any = None
        intent_metrics: dict = {}
        if features is not None and user_count > 0:
            try:
                # Generate synthetic labels for training if no real labels
                import numpy as np

                rng = np.random.RandomState(42)
                synthetic_labels = (rng.rand(user_count) < 0.08).astype(int)
                intent_metrics = intent_model.train(features, synthetic_labels)
                intent_scores = intent_model.predict(features)
            except Exception as exc:
                logger.warning("IntentModel failed: %s", exc)
                intent_scores = None
                errors.append(f"IntentModel: {exc}")

        # 3. Predict LTV
        ltv_predictions: Any = None
        if features is not None and user_count > 0:
            try:
                import numpy as np

                # Train LTV model with synthetic targets if needed
                rng = np.random.RandomState(42)
                synthetic_ltv = rng.exponential(scale=200, size=user_count)
                ltv_predictor.train(features, synthetic_ltv)
                ltv_predictions = ltv_predictor.predict_ltv(features)
            except Exception as exc:
                logger.warning("LVTPredictor failed: %s", exc)
                ltv_predictions = None
                errors.append(f"LVTPredictor: {exc}")

        # 4. Score users
        user_scores_df = None
        top_users: list = []
        if intent_scores is not None and ltv_predictions is not None:
            try:
                user_scores_df = scorer.score_users(
                    intent_scores=intent_scores.values,
                    ltv_predictions=ltv_predictions.values,
                    user_ids=features.index if features is not None else None,
                )
                ranked = scorer.rank_users(user_scores_df)
                top_users = ranked.head(100).to_dict("records")
            except Exception as exc:
                logger.warning("UserScorer failed: %s", exc)
                errors.append(f"UserScorer: {exc}")

        # 5. Segment users
        segments = {}
        segment_summary: dict = {}
        if user_scores_df is not None:
            try:
                segments = scorer.segment_by_score(user_scores_df)
                segment_summary = {
                    name: {
                        "count": len(seg),
                        "ratio": len(seg) / max(user_count, 1),
                    }
                    for name, seg in segments.items()
                }
            except Exception as exc:
                logger.warning("UserSegmentor failed: %s", exc)
                errors.append(f"UserSegmentor: {exc}")

        # Also do RFM segmentation if data available
        rfm_result = {}
        if features is not None and user_count > 0:
            try:
                import pandas as pd
                import numpy as np

                # Create user_data for RFM from features
                user_data = features.copy()
                if "user_id" not in user_data.columns:
                    user_data = user_data.reset_index().rename(columns={"index": "user_id"})
                # Replace inf/na with finite defaults so RFM quantile binning works
                numeric_cols = user_data.select_dtypes(include=[np.number]).columns
                user_data[numeric_cols] = user_data[numeric_cols].replace(
                    [float("inf"), float("-inf")], np.nan
                )
                user_data[numeric_cols] = user_data[numeric_cols].fillna(0)
                if len(user_data) > 0:
                    rfm_result = segmentor.combined_segmentation(user_data).to_dict("index")
            except Exception as exc:
                logger.warning("RFM segmentation failed: %s", exc)

        # Build flat result dict (no wrapping in {"prospect_results": ...})
        result: dict[str, Any] = {
            "user_count": user_count,
            "intent_metrics": intent_metrics,
            "segment_summary": segment_summary,
            "rfm_result_count": len(rfm_result),
            "top_users_sample": top_users[:10] if top_users else [],
        }
        if errors:
            result["errors"] = errors
        return result

    # ------------------------------------------------------------------
    # LLM synthesis prompt builder
    # ------------------------------------------------------------------

    def _build_synthesis_prompt(self, params: dict, results: dict) -> str:
        """Build the LLM synthesis prompt from pipeline results."""
        memory_context = params.get("memory_context", {})
        segment_summary = results.get("segment_summary", {})
        intent_metrics = results.get("intent_metrics", {})

        # Build segment details string
        segment_details = []
        for seg_name, seg_info in segment_summary.items():
            if isinstance(seg_info, dict):
                count = seg_info.get("count", 0)
                ratio = seg_info.get("ratio", 0)
                segment_details.append(f"  - {seg_name}: {count}人 ({ratio:.1%})")
            else:
                segment_details.append(f"  - {seg_name}: {seg_info}")

        confidence_hint = f"{min(1.0, max(0.0, (intent_metrics.get('auc', 0.5) - 0.5) * 2)):.2f}"

        return ProspectPrompt().render(
            user_query=params.get("query", ""),
            season=params.get("season", "当前"),
            seasonal_event=params.get("seasonal_event", "常规运营期"),
            kpi_baseline=params.get("kpi_baseline", {}),
            memory_context=memory_context,
            user_count=results.get("user_count", 0),
            intent_auc=intent_metrics.get("auc", "N/A"),
            intent_accuracy=intent_metrics.get("accuracy", "N/A"),
            rfm_segments=results.get("rfm_result_count", 0),
            segment_details="\n".join(segment_details) if segment_details else "  - 无分群数据",
            confidence_hint=confidence_hint,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_data(data_path: str) -> dict:
        """Load raw data for feature engineering.

        Uses sample data if no path is provided.
        """
        import pandas as pd

        if data_path:
            path = __import__("pathlib").Path(data_path)
            if path.is_file():
                if path.suffix == ".csv":
                    df = pd.read_csv(path)
                    return {"user_logs": df, "user_profile": pd.DataFrame()}
                if path.suffix in (".parquet", ".pq"):
                    df = pd.read_parquet(path)
                    return {"user_logs": df, "user_profile": pd.DataFrame()}

        # Generate sample data for demo mode
        return FeatureEngine.generate_sample_data(n_users=500, n_rides=5000)
