"""SubsidyDeepAgent — Tier 2 补贴策略深度分析 Agent（v10.1 业界水准）。

在 SubsidyExpert 之上加完整 ReAct 推理循环 + ACI 工具规格 + 自检。
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.deep.base import DeepAgentBase
from src.agents.subsidy import SubsidyExpert
from src.core.tool_spec import SUBSIDY_TOOLS

logger = logging.getLogger(__name__)


class SubsidyDeepAgent(DeepAgentBase):
    """Tier 2 补贴策略深度分析 Agent。"""

    def __init__(self, llm=None, max_deep_steps: int = 6, **kw):
        super().__init__(llm=llm, max_deep_steps=max_deep_steps, **kw)
        self._expert = SubsidyExpert(llm=llm)
        self._last_pipeline_result: dict[str, Any] = {}

    @property
    def domain(self) -> str:
        return "subsidy"

    @property
    def tool_specs(self):
        return SUBSIDY_TOOLS

    def _domain_constraints(self) -> list[str]:
        return [
            "补贴金额不能为负",
            "ATE 是因果增量（不是相关性），解释时不得混淆",
            "分配总额不得超出预算",
        ]

    async def _execute_action(self, action: str, params: dict) -> str:
        if action == "finalize":
            return "ready"

        if action == "subsidy_optimize":
            budget = params.get("total_budget_yuan", params.get("budget", 50000))
            result = await self._expert.analyze({"budget": budget})
            self._last_pipeline_result = result
            kpis = result.get("kpis", {})
            return f"ATE={kpis.get('ate', 'N/A')}, allocated={kpis.get('total_allocated', 'N/A')}"

        if action == "elasticity_estimate":
            return "elasticity estimation requires dedicated tool (not yet in pipeline)"

        # 默认: run_pipeline
        result = await self._expert.analyze(params)
        self._last_pipeline_result = result
        kpis = result.get("kpis", {})
        return f"ATE={kpis.get('ate', 'N/A')}, allocated={kpis.get('total_allocated', 'N/A')}"

    async def _compile_final(self, params: dict[str, Any]) -> dict[str, Any]:
        base = self._last_pipeline_result or await self._expert.analyze(params)
        return base
