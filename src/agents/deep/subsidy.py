"""SubsidyDeepAgent — Tier 2 补贴策略深度分析 Agent（PoC）。

在 SubsidyExpert 之上加 ReAct 推理循环：
- 第一轮跑标准管线（因果推断 → 整数规划 → 输出）
- LLM 检查中间结果（ATE 是否合理/弹性是否异常）
- 发现异常 → 调整细分段/参数重跑
- 步数上限 10，幂等缓存防死循环

与 Tier 1 SubsidyExpert 的区别：
| | SubsidyExpert（Tier 1） | SubsidyDeepAgent（Tier 2） |
|---|---|---|
| 执行 | 因果推断→整数规划→LLM综合 | ReAct循环（可迭代参数） |
| LLM | 1次综合 | 3-10次决策+综合 |
| 迭代 | 无 | 可根据ATE结果调整细分段重跑 |
| KPI | 单次输出 | 每步KPI写入L1台账 |
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.deep.base import DeepAgentBase
from src.agents.subsidy import SubsidyExpert

logger = logging.getLogger(__name__)


class SubsidyDeepAgent(DeepAgentBase):
    """Tier 2 补贴策略深度分析 Agent。"""

    def __init__(self, llm=None, max_deep_steps: int = 6, **kw):
        super().__init__(llm=llm, max_deep_steps=max_deep_steps, **kw)
        self._expert = SubsidyExpert(llm=llm)
        self._last_pipeline_result: dict[str, Any] = {}
        self._last_obs: str = ""

    @property
    def domain(self) -> str:
        return "subsidy"

    async def _execute_action(self, action: str, params: dict) -> str:
        """确定性执行——接入 SubsidyExpert 的管线。"""
        if action == "finalize":
            return "ready to compile final result"

        if action == "check_result":
            if not self._last_pipeline_result:
                return "no prior pipeline result — run_pipeline first"
            kpis = self._last_pipeline_result.get("kpis", {})
            ate = kpis.get("ate")
            allocation_count = len(kpis.get("allocation", []))
            return f"ATE={ate:.4f}, allocation_segments={allocation_count}"

        # run_pipeline
        try:
            self._last_pipeline_result = await self._expert.analyze(params)
            kpis = self._last_pipeline_result.get("kpis", {})
            self._last_obs = (
                f"pipeline ok: ATE={kpis.get('ate', 'N/A')}, "
                f"budget_used={kpis.get('total_allocated', 'N/A')}"
            )
            return self._last_obs
        except Exception as exc:
            self._last_obs = f"pipeline failed: {exc}"
            return self._last_obs

    async def _compile_final(self, params: dict) -> dict[str, Any]:
        """编译最终结论——优先用最近一次管线结果 + 步级观测。"""
        base = self._last_pipeline_result or await self._expert.analyze(params)
        base["deep_steps"] = self._deep_steps
        base["deep_step_count"] = len(self._deep_steps)
        base["deep_domain"] = self.domain
        return base
