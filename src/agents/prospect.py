"""ProspectAgent - User prospecting, scoring, segmentation and LTV prediction."""

from __future__ import annotations

import logging
from typing import Any

from src.core.base import BaseAgent
from src.core.state import AgentState

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


SYSTEM_PROMPT = """\
你是 GrowthPilot 潜客识别 Agent。你的职责是：
1. 基于用户行为数据构建特征
2. 预测用户转化意向
3. 对用户进行评分和排序
4. 预测用户生命周期价值 (LTV)
5. 对用户进行分层

请用 JSON 格式输出分析结果。
"""


class ProspectAgent(BaseAgent):
    """Identifies and scores potential high-value users for freight services."""

    name = "prospect"
    description = "潜客识别与评分 Agent"

    def __init__(self, llm: Any, system_prompt: str = SYSTEM_PROMPT) -> None:
        super().__init__(llm=llm, system_prompt=system_prompt)
        self._feature_engine = FeatureEngine()
        self._intent_model = IntentModel()
        self._scorer = UserScorer()
        self._segmentor = UserSegmentor()
        self._ltv_predictor = LVTPredictor()

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Execute the prospect analysis pipeline.

        Returns a partial state update containing ``prospect_results``.
        """
        errors: list[str] = []
        data_path = state.get("data_path", "")

        # 1. Build features
        try:
            raw_data = self._load_data(data_path)
            features = self._feature_engine.build_feature_matrix(raw_data)
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
                intent_metrics = self._intent_model.train(features, synthetic_labels)
                intent_scores = self._intent_model.predict(features)
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
                self._ltv_predictor.train(features, synthetic_ltv)
                ltv_predictions = self._ltv_predictor.predict_ltv(features)
            except Exception as exc:
                logger.warning("LVTPredictor failed: %s", exc)
                ltv_predictions = None
                errors.append(f"LVTPredictor: {exc}")

        # 4. Score users
        user_scores_df = None
        top_users: list = []
        if intent_scores is not None and ltv_predictions is not None:
            try:
                user_scores_df = self._scorer.score_users(
                    intent_scores=intent_scores.values,
                    ltv_predictions=ltv_predictions.values,
                    user_ids=features.index if features is not None else None,
                )
                ranked = self._scorer.rank_users(user_scores_df)
                top_users = ranked.head(100).to_dict("records")
            except Exception as exc:
                logger.warning("UserScorer failed: %s", exc)
                errors.append(f"UserScorer: {exc}")

        # 5. Segment users
        segments = {}
        segment_summary: dict = {}
        if user_scores_df is not None:
            try:
                segments = self._scorer.segment_by_score(user_scores_df)
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

                # Create user_data for RFM from features
                user_data = features.copy()
                if "user_id" not in user_data.columns:
                    user_data = user_data.reset_index().rename(columns={"index": "user_id"})
                rfm_result = self._segmentor.combined_segmentation(user_data).to_dict("index")
            except Exception as exc:
                logger.warning("RFM segmentation failed: %s", exc)

        # 6. LLM synthesis
        try:
            prompt = self._build_prospect_prompt(
                user_count=user_count,
                intent_metrics=intent_metrics,
                segment_summary=segment_summary,
                rfm_segments=len(rfm_result),
                state=state,
            )
            llm_response = await self._invoke_llm(prompt)
            analysis = self._parse_json_response(llm_response)
        except Exception as exc:
            logger.warning("Prospect LLM synthesis failed: %s", exc)
            analysis = {"summary": "LLM synthesis unavailable"}
            errors.append(f"LLM synthesis: {exc}")

        result: dict[str, Any] = {
            "prospect_results": {
                "user_count": user_count,
                "intent_metrics": intent_metrics,
                "segment_summary": segment_summary,
                "rfm_result_count": len(rfm_result),
                "top_users_sample": top_users[:10] if top_users else [],
                "analysis": analysis,
            },
        }
        if errors:
            result["errors"] = errors
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_data(self, data_path: str) -> dict:
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

    def _build_prospect_prompt(
        self,
        *,
        user_count: int,
        intent_metrics: dict,
        segment_summary: dict,
        rfm_segments: int,
        state: AgentState,
        **_: Any,
    ) -> str:
        context = self._build_prompt_context(state)
        return f"""\
{context}

## 潜客识别分析数据

- 用户总数: {user_count}
- 意向模型 AUC: {intent_metrics.get('auc', 'N/A')}
- 意向模型 Accuracy: {intent_metrics.get('accuracy', 'N/A')}
- 用户分层: {segment_summary}
- RFM 分层用户数: {rfm_segments}

请基于以上数据，给出潜客识别的综合分析和策略建议：
1. 高潜用户画像特征
2. 转化意向分析
3. 分层运营建议

请以 JSON 格式输出:
{{
  "summary": "总体概述",
  "high_value_profile": "高价值用户画像",
  "intent_insight": "转化意向洞察",
  "segment_strategy": "分层运营建议"
}}"""
