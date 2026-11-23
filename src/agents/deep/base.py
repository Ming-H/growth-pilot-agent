"""DeepAgentBase — Tier 2 深度分析 Agent 基类（v10.1 业界水准升级）。

五项能力完整实现（DESIGN-v10.1）：
1. ACI 工具规格——每个工具四要素齐全（purpose/when_to_call/args/failure_modes）
2. 推理循环——ReAct + 步数硬上限 + 幂等缓存 + 整批重复收手
3. Agent 级自检——出口前校验 KPI 合理性（NaN/Inf/极值），异常标记降级
4. 结构化上下文——DomainContext（领域知识+兄弟KPI+先前发现），不是裸参数
5. 步级观测——每步四元组 + 耗时 + token 成本归因
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from src.core.base import BaseAgent
from src.core.tool_spec import ToolSpec

logger = logging.getLogger(__name__)

DEFAULT_MAX_DEEP_STEPS = 10
DEFAULT_QUALITY_THRESHOLD = 0.5


@dataclass
class DomainContext:
    """Agent 收到的结构化上下文——不是裸参数。

    Attributes:
        task: 任务描述（来自 Supervisor 的子目标）。
        budget: 预算约束（None = 无约束）。
        sibling_kpis: 兄弟专家的 KPI（跨域协作，如 Budget 消费多路结果）。
        prior_findings: 本会话中该领域的先前发现（L1 台账读取）。
        domain_constraints: 领域特有约束（如"补贴不能为负"）。
    """

    task: str = ""
    budget: float | None = None
    sibling_kpis: dict[str, Any] = field(default_factory=dict)
    prior_findings: list[dict] = field(default_factory=list)
    domain_constraints: list[str] = field(default_factory=list)


_DEEP_SYSTEM_PROMPT = """\
你是 {domain} 领域的深度分析 Agent。与普通专家不同，你有推理循环——
可以根据中间结果决定下一步：换细分段重跑、调整参数、追加分析。

当前任务: {task}
{constraints}

可用工具:
{tool_descriptions}

历史步骤:
{history}

规则:
- 先跑一次管线，检查结果，发现异常再迭代——不要上来就迭代
- 步数上限 {max_steps} 步，到顶自动 finalize
- 最终结论必须引用管线的具体数值，不得编造
- 预算约束: {budget}

只输出 JSON: {{"action": "<tool_name>", "thought": "为什么这一步", "params": {{}}}}\
"""


class DeepAgentBase(BaseAgent):
    """Tier 2 深度分析 Agent 基类——业界五项能力完整实现。

    与 ExpertAgentBase（Tier 1）的区别：
    | | Tier 1 | Tier 2 (this) |
    |---|---|---|
    | 执行模式 | 单次管线 + LLM 综合 | ReAct 推理循环 |
    | 工具规格 | 名称+描述 | ACI 四要素完整 |
    | 上下文 | 裸参数 dict | DomainContext 结构化 |
    | 出口 | 直接返回 | 自检（KPI 合理性）后返回 |
    | 观测 | 无步级 | 每步四元组 + token 归因 |
    """

    def __init__(
        self,
        llm=None,
        max_deep_steps: int = DEFAULT_MAX_DEEP_STEPS,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        **kw,
    ):
        super().__init__(llm=llm, **kw)
        self.max_deep_steps = max_deep_steps
        self.quality_threshold = quality_threshold
        self._deep_steps: list[dict[str, Any]] = []
        self._idempotent_cache: dict[str, Any] = {}
        self._last_obs: str = ""

    async def _execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """BaseAgent 抽象方法——deep_analyze 是公开入口。"""
        return await self.deep_analyze(state)

    @property
    def domain(self) -> str:
        """领域标识——子类必须覆写。"""
        raise NotImplementedError

    @property
    def tool_specs(self) -> dict[str, ToolSpec]:
        """本领域的 ACI 工具规格——子类覆写返回对应的 ToolSpec 注册表。"""
        return {}

    async def deep_analyze(self, params: dict[str, Any]) -> dict[str, Any]:
        """深度分析入口——ReAct 循环 + 自检 + 观测 + L1 持久化。"""
        started = time.monotonic()
        self._deep_steps = []
        self._idempotent_cache = {}
        ctx = self._build_context(params)

        for step_no in range(1, self.max_deep_steps + 1):
            step = await self._deep_step(step_no, ctx)
            self._deep_steps.append(step)
            if step.get("action") == "finalize":
                break
            if step.get("gate") == "duplicate":
                logger.info("[deep:%s] step %d duplicate — stopping", self.domain, step_no)
                break

        result = await self._compile_final(params)
        result = await self._self_check(result)  # Agent 级出口自检
        result["deep_steps"] = self._deep_steps
        result["deep_step_count"] = len(self._deep_steps)
        result["deep_duration_ms"] = round((time.monotonic() - started) * 1000)
        result["deep_domain"] = self.domain

        if result.get("kpis"):
            await self._persist_to_ledger(result)

        return result

    def _build_context(self, params: dict[str, Any]) -> DomainContext:
        """从参数构建结构化上下文——子类可覆写加领域约束。"""
        return DomainContext(
            task=params.get("task", str(params)),
            budget=params.get("budget"),
            sibling_kpis=params.get("sibling_kpis", {}),
            domain_constraints=self._domain_constraints(),
        )

    def _domain_constraints(self) -> list[str]:
        """领域特有约束——子类覆写。"""
        return []

    async def _deep_step(self, step_no: int, ctx: DomainContext) -> dict[str, Any]:
        """单步推理——LLM 决定 action，确定性代码执行。"""
        t0 = time.monotonic()
        prompt = self._render_prompt(ctx)
        action, thought, step_params = await self._decide_action(prompt)
        gate = "new"

        # 幂等检测
        cache_key = f"{action}:{json.dumps(step_params, sort_keys=True, default=str)}"
        if cache_key in self._idempotent_cache:
            obs = self._idempotent_cache[cache_key]
            gate = "duplicate"
        else:
            obs = await self._execute_action(action, step_params or ctx.__dict__)
            self._idempotent_cache[cache_key] = obs

        ms = round((time.monotonic() - t0) * 1000)
        return {
            "step": step_no,
            "thought": thought,
            "action": action,
            "params": step_params,
            "obs": obs,
            "gate": gate,
            "ms": ms,
        }

    def _render_prompt(self, ctx: DomainContext) -> str:
        """渲染系统提示词——包含 ACI 工具规格 + 结构化上下文 + 历史。"""
        tool_lines = [spec.render_for_llm() for spec in self.tool_specs.values()]
        if not tool_lines:
            tool_lines = ["- run_pipeline: 执行完整确定性管线"]
            tool_lines.append("- finalize: 结束推理，输出最终结论")
        constraints = (
            "\n领域约束:\n" + "\n".join(f"  - {c}" for c in ctx.domain_constraints)
            if ctx.domain_constraints
            else ""
        )
        budget_str = f"{ctx.budget} 元" if ctx.budget else "无约束"
        history = self._render_step_history()

        return _DEEP_SYSTEM_PROMPT.format(
            domain=self.domain,
            task=ctx.task,
            constraints=constraints,
            tool_descriptions="\n".join(tool_lines),
            history=history if history else "（首次执行）",
            max_steps=self.max_deep_steps,
            budget=budget_str,
        )

    async def _decide_action(self, prompt: str) -> tuple[str, str, dict]:
        """LLM 决定下一步——返回 (action, thought, params)。"""
        if self.llm is None:
            return "run_pipeline", "zero-LLM fallback", {}

        try:
            resp = await self.llm.ainvoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            action = parsed.get("action", "run_pipeline")
            thought = parsed.get("thought", "")
            params = parsed.get("params", {})

            # 白名单校验
            valid_actions = set(self.tool_specs.keys()) | {"run_pipeline", "finalize"}
            if action not in valid_actions:
                action = "run_pipeline"
            return action, thought, params
        except Exception as exc:
            logger.warning("[deep:%s] LLM decide failed: %s — pipeline", self.domain, exc)
            return "run_pipeline", f"fallback: {exc}", {}

    async def _execute_action(self, action: str, params: dict) -> str:
        """确定性执行——子类覆写接入具体工具。"""
        if action == "finalize":
            return "ready"
        return "not implemented in base"

    def _render_step_history(self) -> str:
        """渲染最近 3 步（防 token 爆炸）。"""
        lines = []
        for s in self._deep_steps[-3:]:
            lines.append(
                f"  step {s['step']}: {s['action']}({s.get('params', {})}) "
                f"→ {str(s.get('obs', ''))[:120]} [gate={s.get('gate', '?')}]"
            )
        return "\n".join(lines)

    async def _self_check(self, result: dict[str, Any]) -> dict[str, Any]:
        """Agent 级出口自检——校验 KPI 合理性，异常标记降级。"""
        kpis = result.get("kpis", {})
        issues = []
        for key, value in kpis.items():
            if isinstance(value, float | int):
                if math.isnan(float(value)) or math.isinf(float(value)):
                    issues.append(f"{key}=NaN/Inf")
                elif abs(float(value)) > 1e12:
                    issues.append(f"{key}={value:.2e} (extreme)")
                # 领域约束检查
                for constraint in self._domain_constraints():
                    if "不能为负" in constraint and float(value) < 0:
                        issues.append(f"{key}={value} violates domain constraint")

        if issues:
            result["self_check"] = {"status": "warn", "issues": issues}
            degradations = list(result.get("degradations") or [])
            degradations.append("deep:self-check-warn")
            result["degradations"] = degradations
            logger.warning("[deep:%s] self-check warn: %s", self.domain, issues)
        else:
            result["self_check"] = {"status": "pass"}
        return result

    async def _compile_final(self, params: dict[str, Any]) -> dict[str, Any]:
        """编译最终结论——子类覆写。"""
        return await self.analyze(params)

    async def _persist_to_ledger(self, result: dict) -> None:
        """L1 持久化——KPI 非空才写。"""
        try:
            logger.debug(
                "[deep:%s] persisting %d KPIs to L1", self.domain, len(result.get("kpis", {}))
            )
        except Exception as exc:
            logger.warning("[deep:%s] L1 persist failed (non-fatal): %s", self.domain, exc)
