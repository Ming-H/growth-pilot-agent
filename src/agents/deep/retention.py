"""RetentionDeepAgent — Tier 2 留存召回深度分析 Agent。

在 RetentionExpert 之上加 ReAct 推理循环：可迭代调整流失阈值/同期群窗口/召回策略。
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.deep.base import DeepAgentBase
from src.agents.retention import RetentionExpert
from src.core.tool_spec import RETENTION_TOOLS

logger = logging.getLogger(__name__)


class RetentionDeepAgent(DeepAgentBase):
    """Tier 2 留存召回深度分析 Agent。"""

    def __init__(self, llm=None, max_deep_steps: int = 5, **kw):
        super().__init__(llm=llm, max_deep_steps=max_deep_steps, **kw)
        self._expert = RetentionExpert(llm=llm)
        self._last_pipeline_result: dict[str, Any] = {}

    @property
    def domain(self) -> str:
        return "retention"

    @property
    def tool_specs(self):
        return RETENTION_TOOLS

    def _domain_constraints(self) -> list[str]:
        return [
            "流失概率范围 [0, 1]",
            "churn_high_risk_count 是 KPI（用于闸门），不是 len(high_risk_users)",
            "同期群分析结果需标注观察窗口",
        ]

    async def _execute_action(self, action: str, params: dict) -> str:
        if action == "finalize":
            return "ready"

        if action == "churn_predict":
            result = await self._expert.analyze(params)
        elif action == "cohort_analyze":
            result = await self._expert.analyze(params)
        else:
            result = await self._expert.analyze(params)

        self._last_pipeline_result = result
        kpis = result.get("kpis", {})
        return f"churn_rate={kpis.get('churn_rate', 'N/A')}, high_risk={kpis.get('churn_high_risk_count', 'N/A')}"

    async def _compile_final(self, params: dict[str, Any]) -> dict[str, Any]:
        base = self._last_pipeline_result or await self._expert.analyze(params)
        return base
