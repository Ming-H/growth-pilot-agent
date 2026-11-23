"""Supervisor 节点——分层架构的编排入口。

查询理解 → 复杂度判定（确定性规则 + LLM 灰色地带） → 路由到对应层级。

设计（DESIGN-v10 §4）：
- 确定性规则覆盖 >70% 场景（`tier.determine_tier`）
- 灰色地带走 LLM 判定（role=fast, 单次调用, 3 选 1 分类）
- 零 LLM 时 Supervisor 完全退化为确定性规则（跳过灰色地带）
- 层级判定结果写入 state，全部降级可见

与 v9 兼容：
- Tier.WORKFLOW → 现有 workflow 模式（`src/graph/graph.py`）
- Tier.EXPERT   → 现有 agent 模式（`src/agent/graph.py`）
- Tier.DEEP     → 新增 deep 模式（`src/agents/deep/`）
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.supervisor.tier import Tier, count_domains, determine_tier, is_factoid

logger = logging.getLogger(__name__)

# 灰色地带 LLM 判定提示词（3 选 1 分类，role=fast 足够）
_TIER_CLASSIFY_PROMPT = """\
你是增长分析系统的调度器。根据用户查询，判定应使用哪个分析层级：

- tier 0（workflow）: 简单事实查询，确定性管线足够（"XX 的 AUC 是多少"）
- tier 1（expert）: 单领域分析，专家 Agent 做一次综合（"帮我优化补贴策略"）
- tier 2（deep）: 跨域迭代分析，需要多步推理（"对比六种策略并给出全局预算分配"）

用户查询: {query}
预算: {budget}
命中领域数: {domains}

规则:
- 拿不准打 1（默认中间档，性价比最优）
- 涉及跨域比较、全局优化、迭代调整的打 2
- 纯事实查询（查一个数/一个列表）打 0

只输出 JSON: {{"tier": 0, "reason": "一句话"}}\
"""


async def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Supervisor 节点：复杂度判定 + 层级路由。

    写入 state:
        tier: Tier 枚举值（0/1/2）
        tier_reason: 判定依据（一句话）
        tier_source: "rule"（确定性规则）| "llm"（灰色地带 LLM）| "forced"
        tier_domains: 命中的领域列表
        degradations: 追加 "supervisor:no-llm" 当灰色地带无法判定时
    """
    started = time.monotonic()
    query = state.get("query", "")
    budget = state.get("budget")

    # 1) 确定性规则判定（覆盖 >70% 场景）
    ctx = {
        "mode": state.get("_ci_mode"),
        "force_tier": state.get("_force_tier"),
    }
    tier = determine_tier(query, budget, ctx)
    source = "rule"

    # 2) 灰色地带 → LLM 判定（可选）
    degradations: list[str] = list(state.get("degradations") or [])
    if tier == Tier.EXPERT and _should_try_llm(state):
        llm_tier = await _llm_classify(query, budget, count_domains(query), state)
        if llm_tier is not None:
            tier = llm_tier
            source = "llm"
        else:
            degradations.append("supervisor:rule-only")

    # 3) 写入 state
    dur_ms = round((time.monotonic() - started) * 1000)
    logger.info(
        "[supervisor] tier=%s source=%s domains=%d factoid=%s (%dms)",
        tier.label,
        source,
        count_domains(query),
        is_factoid(query),
        dur_ms,
    )
    return {
        "tier": int(tier),
        "tier_label": tier.label,
        "tier_source": source,
        "tier_reason": (
            f"domains={count_domains(query)} factoid={is_factoid(query)} budget={budget}"
        ),
        "tier_domains": sorted(_hit_domains(query)),
        "degradations": degradations,
    }


def _should_try_llm(state: dict[str, Any]) -> bool:
    """灰色地带是否尝试 LLM 判定——需要 LLM 可用且非 CI 模式。"""
    if state.get("_ci_mode"):
        return False
    from src.core.config import get_settings

    s = get_settings()
    if s.llm_channel == "local" and not s.llm_api_key:
        return True  # 本地免 key
    return bool(s.llm_api_key)


async def _llm_classify(query: str, budget, domains: int, state) -> Tier | None:
    """灰色地带 LLM 判定（role=fast, 单次调用）。失败返回 None。"""
    try:
        from src.core.llm_factory import create_llm

        llm = create_llm(role="fast", agent_name="supervisor")
        if llm is None:
            return None
        prompt = _TIER_CLASSIFY_PROMPT.format(
            query=query, budget=budget or "未指定", domains=domains
        )
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)
        return Tier(int(parsed.get("tier", 1)))
    except Exception as exc:
        logger.warning("[supervisor] LLM classify failed, fallback to rule: %s", exc)
        return None


def _hit_domains(query: str) -> set[str]:
    """命中领域的集合（用于 state 记录）。"""
    from src.supervisor.tier import _DOMAIN_KEYWORDS

    hits = set()
    for domain, words in _DOMAIN_KEYWORDS.items():
        if any(w in query for w in words):
            hits.add(domain)
    return hits
