"""ToolSpec — Agent-Computer Interface 数据结构。

Anthropic《Building Effective Agents》(2024.12) 第 4 条原则：
"Tool design and selection are critical. Agent-tool interfaces are as
critical as human-computer interfaces."

每个工具的 ACI 四要素：purpose / when_to_call / args / failure_modes。
LLM 看到的不是"名字+描述"，是完整的工具画像——知道什么时候调、
怎么调、调了会返回什么、会怎么失败。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolSpec:
    """Agent-Computer Interface——LLM 看到的工具完整画像。

    Attributes:
        name: 工具标识（与函数名一致）。
        purpose: 做什么（一句话，给 LLM 的第一印象）。
        when_to_call: 什么时候该调这个工具（vs 其他工具的选择依据）。
        args: 参数说明（参数名 → 用途描述）。
        failure_modes: 已知失败模式（LLM 需要知道才能规避）。
        output_schema: 输出结构描述（key → 类型+含义）。
        cost_hint: 成本提示（"轻量"/"需要训练模型，较慢"/"内存密集"）。
    """

    name: str
    purpose: str
    when_to_call: str
    args: dict[str, str] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    output_schema: dict[str, str] = field(default_factory=dict)
    cost_hint: str = "轻量"

    def render_for_llm(self) -> str:
        """渲染为 LLM 可读的工具描述（嵌入系统提示词）。"""
        lines = [f"- {self.name}: {self.purpose}"]
        lines.append(f"  调用时机: {self.when_to_call}")
        if self.args:
            arg_str = "; ".join(f"{k}={v}" for k, v in self.args.items())
            lines.append(f"  参数: {arg_str}")
        if self.failure_modes:
            lines.append(f"  已知失败: {'; '.join(self.failure_modes)}")
        if self.cost_hint != "轻量":
            lines.append(f"  成本: {self.cost_hint}")
        return "\n".join(lines)


def register_tools(*specs: ToolSpec) -> dict[str, ToolSpec]:
    """注册工具规格——返回 name → ToolSpec 映射。"""
    return {s.name: s for s in specs}


# ── 现有工具的 ACI 规格（按领域分组）────────────────────────────────────────


SUBSIDY_TOOLS = register_tools(
    ToolSpec(
        name="subsidy_optimize",
        purpose="ATE 因果估计 + 整数规划优惠券分配（给定预算求全局最优发券策略）",
        when_to_call="用户问补贴策略/发券 ROI/预算分配时；需要因果增量而非相关性的场景",
        args={
            "total_budget_yuan": "总预算（元），必须为正数",
            "segment_count": "细分段数（可选，默认全量）",
        },
        failure_modes=[
            "预算 ≤0 时返回空分配（不是报错，是合法空解）",
            "因果引擎在极端不平衡数据上 ATE 可能接近 0（无增量）",
        ],
        output_schema={
            "ate": "float — 平均处理效应（因果增量转化率）",
            "allocation": "list[dict] — 各细分段的分配额与预期增量",
            "total_allocated": "float — 实际分配总额（≤ 预算）",
        },
        cost_hint="需要训练因果模型，首次较慢（后续有缓存）",
    ),
    ToolSpec(
        name="elasticity_estimate",
        purpose="价格弹性估计（不同价格水平对需求的边际影响）",
        when_to_call="需要理解价格敏感度/优化定价时；与 subsidy_optimize 配合使用",
        args={"price_range": "价格区间（可选，默认自动）"},
        failure_modes=["数据量不足时弹性估计置信区间很宽"],
        output_schema={"elasticity": "float — 价格弹性系数", "confidence": "float — 置信度"},
    ),
)

PROSPECT_TOOLS = register_tools(
    ToolSpec(
        name="intent_predict",
        purpose="用户购车意向预测（0-100 评分）",
        when_to_call="需要识别高潜用户/圈选目标人群时",
        args={"user_ids": "用户 ID 列表（可选，默认全量）"},
        failure_modes=["冷启动用户（无历史行为）预测置信度低"],
        output_schema={"scores": "list[dict] — 用户ID+意向分+置信度", "auc": "float — 模型 AUC"},
    ),
    ToolSpec(
        name="ltv_rank",
        purpose="用户生命周期价值（LTV）排序",
        when_to_call="需要按长期价值排优先级/分配资源时",
        args={"top_k": "返回 Top-K 用户（默认 100）"},
        output_schema={"ranked_users": "list[dict] — 用户ID+LTV预测值", "mean_ltv": "float"},
    ),
)

RETENTION_TOOLS = register_tools(
    ToolSpec(
        name="churn_predict",
        purpose="流失预警（预测哪些用户即将流失）",
        when_to_call="用户问流失率/预警名单/挽留策略时",
        args={"threshold": "流失概率阈值（默认 0.7）"},
        failure_modes=["新用户（<7 天）无足够行为数据，预测不准确"],
        output_schema={
            "high_risk_users": "list[dict] — 流失概率>阈值的用户",
            "auc": "float — 模型 AUC",
            "churn_rate": "float — 整体流失率",
        },
    ),
    ToolSpec(
        name="cohort_analyze",
        purpose="同期群分析（按注册/激活时间分组的留存对比）",
        when_to_call="需要看不同时期用户的留存差异时",
        args={"cohort_period": "分组周期（day/week/month）"},
        output_schema={"cohorts": "list[dict] — 各同期群的留存曲线", "trend": "str — 趋势判断"},
    ),
)

# 全量注册表（DeepAgent 按领域取用）
ALL_TOOL_SPECS: dict[str, ToolSpec] = {**SUBSIDY_TOOLS, **PROSPECT_TOOLS, **RETENTION_TOOLS}
