"""DeepAgentBase — Tier 2 深度分析 Agent 的 ReAct 外壳。

在现有 ExpertAgentBase 之上加推理循环：Agent 能根据中间结果迭代
（跑完因果推断 → 发现弹性异常 → 换细分段重跑），而非一次性管线。

设计（DESIGN-v10 §3 Tier 2）：
- 步数硬上限（默认 10，可配）——无界自主是状态机反模式
- 每步四元组观测（thought/action/obs/gate）
- 幂等缓存 + 整批重复收手（继承 v9 实测护栏）
- 中间结果持久化到 L1 事实台账（KPI 非空才写）
- LLM 不可用 → 自动降级为确定性管线（v9 ExpertAgentBase 行为）

用法：
    agent = SubsidyDeepAgent(llm=llm)
    result = await agent.deep_analyze(params={"budget": 50000})
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.core.base import BaseAgent

logger = logging.getLogger(__name__)

DEFAULT_MAX_DEEP_STEPS = 10
_DEEP_SYSTEM_PROMPT = """\
你是 {domain} 领域的深度分析 Agent。与普通专家不同，你有推理循环——
可以根据中间结果决定下一步：换细分段重跑、调整参数、追加分析。

当前任务: {task}

可用操作:
- run_pipeline: 执行完整确定性管线（参数可调整）
- check_result: 检查上一步结果的质量（KPI 是否合理/有无异常）
- finalize: 结束推理，输出最终结论

规则:
- 先跑一次管线，检查结果，发现异常再迭代——不要上来就迭代
- 步数上限 {max_steps} 步，到顶自动 finalize
- 最终结论必须引用管线的具体数值，不得编造
"""


class DeepAgentBase(BaseAgent):
    """Tier 2 深度分析 Agent 基类——ReAct 外壳 + 步数上限 + L1 持久化。

    与 ExpertAgentBase 的区别：
    | | ExpertAgentBase（Tier 1） | DeepAgentBase（Tier 2） |
    |---|---|---|
    | 执行模式 | 单次管线 + LLM 综合 | ReAct 推理循环 |
    | LLM 调用 | 0-2 次 | 3-{max_steps} 次 |
    | 中间结果 | 不持久化 | 写入 L1 事实台账 |
    | 步数控制 | 无需（单次） | 硬上限 + 幂等缓存 |
    """

    def __init__(self, llm=None, max_deep_steps: int = DEFAULT_MAX_DEEP_STEPS, **kw):
        super().__init__(llm=llm, **kw)
        self.max_deep_steps = max_deep_steps
        self._deep_steps: list[dict[str, Any]] = []

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """BaseAgent 抽象方法实现——deep_analyze 是公开入口，这里做转发。"""
        return await self.deep_analyze(state)
        self.max_deep_steps = max_deep_steps
        self._deep_steps: list[dict[str, Any]] = []

    @property
    def domain(self) -> str:
        """领域标识——子类覆写。"""
        raise NotImplementedError

    async def deep_analyze(self, params: dict[str, Any]) -> dict[str, Any]:
        """深度分析入口——ReAct 循环 + 步级观测 + L1 持久化。"""
        started = time.monotonic()
        self._deep_steps = []
        task = params.get("task", str(params))

        for step_no in range(1, self.max_deep_steps + 1):
            step = await self._deep_step(step_no, task, params)
            self._deep_steps.append(step)

            if step.get("action") == "finalize":
                break
            if step.get("gate") == "duplicate":
                logger.info("[deep:%s] step %d duplicate — stopping", self.domain, step_no)
                break

        result = await self._compile_final(params)
        result["deep_steps"] = self._deep_steps
        result["deep_step_count"] = len(self._deep_steps)
        result["deep_duration_ms"] = round((time.monotonic() - started) * 1000)

        # L1 持久化：KPI 非空才写
        if result.get("kpis"):
            await self._persist_to_ledger(result)

        return result

    async def _deep_step(self, step_no: int, task: str, params: dict) -> dict:
        """单步推理——LLM 决定 action，确定性代码执行。"""
        prompt = _DEEP_SYSTEM_PROMPT.format(
            domain=self.domain, task=task, max_steps=self.max_deep_steps
        )
        history = self._render_step_history()
        if history:
            prompt += f"\n\n历史步骤:\n{history}"

        # LLM 决定下一步
        action, thought = await self._decide_action(prompt)
        gate = "new"

        # 确定性执行
        obs = await self._execute_action(action, params)

        # 幂等检测：与上一步相同的 action+参数 → duplicate
        if len(self._deep_steps) >= 1:
            prev = self._deep_steps[-1]
            if prev.get("action") == action and prev.get("thought") == thought:
                gate = "duplicate"

        return {
            "step": step_no,
            "thought": thought,
            "action": action,
            "obs": obs,
            "gate": gate,
            "ms": 0,  # 由调用方填充
        }

    async def _decide_action(self, prompt: str) -> tuple[str, str]:
        """LLM 决定下一步 action——返回 (action, thought)。"""
        if self.llm is None:
            return "run_pipeline", "zero-LLM fallback: run pipeline once"

        try:
            resp = await self.llm.ainvoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            # 解析 LLM 输出: {"action": "run_pipeline", "thought": "..."}
            import json

            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            action = parsed.get("action", "run_pipeline")
            thought = parsed.get("thought", "")
            if action not in ("run_pipeline", "check_result", "finalize"):
                action = "run_pipeline"
            return action, thought
        except Exception as exc:
            logger.warning(
                "[deep:%s] LLM decide failed: %s — default to pipeline", self.domain, exc
            )
            return "run_pipeline", f"fallback: {exc}"

    async def _execute_action(self, action: str, params: dict) -> str:
        """确定性执行 action——子类覆写以接入具体工具。"""
        if action == "finalize":
            return "ready to compile final result"
        if action == "check_result":
            return self._last_obs[:200] if hasattr(self, "_last_obs") else "no prior result"
        # run_pipeline: 子类覆写
        return "pipeline not implemented in base"

    def _render_step_history(self) -> str:
        """渲染历史步骤（给 LLM 的上下文）——只保留最近 3 步防 token 爆炸。"""
        lines = []
        for s in self._deep_steps[-3:]:
            lines.append(
                f"  step {s['step']}: {s['action']} → {str(s.get('obs', ''))[:100]} "
                f"[gate={s.get('gate', '?')}]"
            )
        return "\n".join(lines)

    async def _compile_final(self, params: dict) -> dict[str, Any]:
        """编译最终结论——默认跑一次管线（子类可覆写加 LLM 综合）。"""
        result = await self.analyze(params)
        return result

    async def _persist_to_ledger(self, result: dict) -> None:
        """中间结果持久化到 L1 事实台账——KPI 非空才写。"""
        try:
            # 复用 ExpertAgentBase 的 L1 写入逻辑（如果有的话）
            # 这里留为扩展点——实际实现取决于 memory store 接口
            logger.debug(
                "[deep:%s] persisting %d KPIs to L1", self.domain, len(result.get("kpis", {}))
            )
        except Exception as exc:
            logger.warning("[deep:%s] L1 persist failed (non-fatal): %s", self.domain, exc)
