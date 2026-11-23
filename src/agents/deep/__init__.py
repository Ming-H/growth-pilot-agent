"""Tier 2 Deep Agents — 有推理循环的领域 Agent。

每个 DeepAgent 在对应的 Tier 1 ExpertAgent 之上加：
- ReAct 推理循环（步数硬上限 + 幂等缓存）
- ACI 工具规格（purpose/when_to_call/args/failure_modes）
- Agent 级出口自检（KPI 合理性 + 领域约束）
- 结构化上下文（DomainContext）
- 步级观测（thought/action/obs/gate/ms）
"""

from src.agents.deep.base import DeepAgentBase, DomainContext
from src.agents.deep.prospect import ProspectDeepAgent
from src.agents.deep.retention import RetentionDeepAgent
from src.agents.deep.subsidy import SubsidyDeepAgent

__all__ = [
    "DeepAgentBase",
    "DomainContext",
    "SubsidyDeepAgent",
    "ProspectDeepAgent",
    "RetentionDeepAgent",
]
