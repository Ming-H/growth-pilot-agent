"""Tier 枚举 + 复杂度判定（纯函数，零 LLM，CI 安全）。

分层架构的核心决策模块：给定查询 + 预算 + 上下文，判定应走哪一层。
确定性规则优先（覆盖 >70% 场景），LLM 判定仅用于灰色地带。

设计（DESIGN-v10 §4.1）：
- 硬规则 1: 预算不足 → Tier 0（不烧 token）
- 硬规则 2: CI/回归/监控 → Tier 0
- 硬规则 3: 显式指定 → 尊重调用方
- 硬规则 4: 跨域 ≥3 + 高预算 → Tier 2
- 硬规则 5: 单领域事实查询 → Tier 0
- 灰色地带 → 由调用方决定是否走 LLM 判定（本模块不做 LLM 调用）
"""

from __future__ import annotations

import re
from enum import IntEnum
from typing import Any


class Tier(IntEnum):
    """调度层级——值越大自主性越高、token 成本越高。"""

    WORKFLOW = 0  # 确定性管线（零 LLM）
    EXPERT = 1  # 专家 Agent（1-2 次 LLM）
    DEEP = 2  # 深度分析 Agent（3-10 次 LLM）

    @property
    def label(self) -> str:
        return {0: "workflow", 1: "expert", 2: "deep"}[self.value]


# ── 领域关键词（与 _intent_classify 共享词汇表）─────────────────────────────

_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "prospect": ("潜客", "高潜", "意向", "拉新", "获客", "新客", "LTV", "评分", "人群"),
    "conversion": ("转化", "漏斗", "优惠券", "触达", "归因", "转化率", "承接"),
    "subsidy": ("补贴", "发券", "ROI", "因果", "增量", "弹性", "ATE"),
    "retention": ("流失", "留存", "召回", "churn", "同期群", "回访"),
    "ad": ("广告", "投放", "RTA", "oCPC", "创意", "竞价", "CPM"),
    "budget": ("预算", "分配", "统筹", "跨渠道", "全局"),
}

_FACTOID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(多少|是什么|是几|哪个|哪些|列[一出]|列出)"),
    re.compile(r"\b(AUC|ROI|CPA|GMV|LTV|ARPU|CTR|CVR)\b", re.IGNORECASE),
    re.compile(r"(Top\s*\d+|前\s*\d+|最高|最低|最大|最小)"),
)


def count_domains(query: str) -> int:
    """查询命中多少个领域——跨域越多越可能需要深度分析。"""
    hits = set()
    for domain, words in _DOMAIN_KEYWORDS.items():
        if any(w in query for w in words):
            hits.add(domain)
    return len(hits)


def is_factoid(query: str) -> bool:
    """是否为事实型查询（只需要查一个数/一个列表，不需要分析推理）。"""
    return any(p.search(query) for p in _FACTOID_PATTERNS)


def _in_explicit_domains(query: str) -> set[str]:
    """命中的具体领域集合（供 Supervisor 记录到 state）。"""
    hits = set()
    for domain, words in _DOMAIN_KEYWORDS.items():
        if any(w in query for w in words):
            hits.add(domain)
    return hits


def determine_tier(
    query: str,
    budget: float | None = None,
    context: dict[str, Any] | None = None,
    *,
    tier0_budget_threshold: float = 500.0,
    tier2_budget_threshold: float = 10000.0,
    tier2_domain_threshold: int = 3,
) -> Tier:
    """确定性判定查询应走哪一层。

    本函数是纯函数（零 LLM 调用），覆盖 >70% 的场景。
    灰色地带返回 Tier.EXPERT 作为安全默认值——由调用方决定
    是否追加 LLM 判定（supervisor.node 的灰色地带通道）。

    Args:
        query: 用户查询文本。
        budget: 预算（元）；None 表示未指定。
        context: 附加上下文（mode/force_tier 等）。
        tier0_budget_threshold: 低于此预算走 Tier 0。
        tier2_budget_threshold: 高于此预算才可能走 Tier 2。
        tier2_domain_threshold: 跨域数达到此值才可能走 Tier 2。

    Returns:
        推荐层级。
    """
    ctx = context or {}

    # 硬规则 1: 预算不足 → Tier 0
    if budget is not None and budget < tier0_budget_threshold:
        return Tier.WORKFLOW

    # 硬规则 2: CI/回归/监控 → Tier 0
    if ctx.get("mode") in ("regression", "ci", "monitoring"):
        return Tier.WORKFLOW

    # 硬规则 3: 显式指定 → 尊重调用方
    forced = ctx.get("force_tier")
    if forced is not None:
        try:
            return Tier(int(forced))
        except (ValueError, TypeError):
            pass  # 非法值走默认逻辑

    domains = count_domains(query)

    # 硬规则 4: 跨域 + 高预算 → Tier 2
    if domains >= tier2_domain_threshold and (
        budget is not None and budget >= tier2_budget_threshold
    ):
        return Tier.DEEP

    # 硬规则 5: 单领域事实查询 → Tier 0
    if domains <= 1 and is_factoid(query):
        return Tier.WORKFLOW

    # 灰色地带 → 默认 Tier 1（性价比最优的中间档）
    return Tier.EXPERT


def tier_degradation_chain(from_tier: Tier) -> list[Tier]:
    """从指定层级开始的降级链（含自身）。"""
    return [Tier(v) for v in range(from_tier.value, -1, -1)]
