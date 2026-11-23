"""ProspectDeepAgent — Tier 2 潜客识别深度分析 Agent。

在 ProspectExpert 之上加 ReAct 推理循环：可迭代调整评分阈值/分群参数/排序策略。
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.deep.base import DeepAgentBase
from src.agents.prospect import ProspectExpert
from src.core.tool_spec import PROSPECT_TOOLS

logger = logging.getLogger(__name__)


class ProspectDeepAgent(DeepAgentBase):
    """Tier 2 潜客识别深度分析 Agent。"""

    def __init__(self, llm=None, max_deep_steps: int = 5, **kw):
        super().__init__(llm=llm, max_deep_steps=max_deep_steps, **kw)
        self._expert = ProspectExpert(llm=llm)
        self._last_pipeline_result: dict[str, Any] = {}

    @property
    def domain(self) -> str:
        return "prospect"

    @property
    def tool_specs(self):
        return PROSPECT_TOOLS

    def _domain_constraints(self) -> list[str]:
        return [
            "意向评分范围 [0, 100]，超出即为异常",
            "LTV 预测值不能为负",
            "冷启动用户（无历史行为）置信度标记为 low",
        ]

    async def _execute_action(self, action: str, params: dict) -> str:
        if action == "finalize":
            return "ready"

        if action == "intent_predict":
            result = await self._expert.analyze(params)
        elif action == "ltv_rank":
            result = await self._expert.analyze(params)
        else:
            result = await self._expert.analyze(params)

        self._last_pipeline_result = result
        kpis = result.get("kpis", {})
        return (
            f"intent_scores={len(kpis.get('scores', []))}, mean_ltv={kpis.get('mean_ltv', 'N/A')}"
        )

    async def _compile_final(self, params: dict[str, Any]) -> dict[str, Any]:
        base = self._last_pipeline_result or await self._expert.analyze(params)
        return base
