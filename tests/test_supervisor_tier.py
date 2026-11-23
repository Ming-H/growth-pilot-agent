"""Supervisor 层级判定测试（纯函数，零 LLM，CI 安全）。

覆盖：硬规则（预算/CI/跨域/事实查询）+ 分级语义 + 降级链。
"""

from __future__ import annotations

from src.supervisor.tier import (
    Tier,
    count_domains,
    determine_tier,
    is_factoid,
    tier_degradation_chain,
)

# ── count_domains ─────────────────────────────────────────────────────────


def test_count_domains_single():
    assert count_domains("帮我优化补贴策略") == 1


def test_count_domains_multi():
    assert count_domains("补贴策略对流失用户的影响") >= 2  # subsidy + retention


def test_count_domains_none():
    assert count_domains("今天天气怎么样") == 0


# ── is_factoid ────────────────────────────────────────────────────────────


def test_is_factoid_true():
    assert is_factoid("AUC 是多少")
    assert is_factoid("ROI 是什么")
    assert is_factoid("列出 Top 10 高潜用户")


def test_is_factoid_false():
    assert not is_factoid("帮我优化补贴策略并给出建议")
    assert not is_factoid("对比不同渠道的转化效果")


# ── determine_tier: 硬规则 ────────────────────────────────────────────────


def test_low_budget_forces_tier0():
    assert determine_tier("帮我做全局分析", budget=100.0) == Tier.WORKFLOW


def test_ci_mode_forces_tier0():
    assert determine_tier("分析", context={"mode": "regression"}) == Tier.WORKFLOW


def test_forced_tier_respected():
    assert determine_tier("简单查询", context={"force_tier": 2}) == Tier.DEEP


def test_forced_invalid_tier_falls_through():
    # 非法 force_tier 值走默认逻辑
    result = determine_tier("帮我优化补贴", context={"force_tier": "invalid"})
    assert result in (Tier.WORKFLOW, Tier.EXPERT, Tier.DEEP)


def test_multi_domain_high_budget_tier2():
    assert (
        determine_tier(
            "对比补贴留存广告投放效果并给出全局预算分配",
            budget=50000.0,
        )
        == Tier.DEEP
    )


def test_single_domain_factoid_tier0():
    assert determine_tier("补贴 ROI 是多少") == Tier.WORKFLOW


def test_analytical_query_defaults_tier1():
    assert determine_tier("帮我优化补贴策略", budget=5000.0) == Tier.EXPERT


def test_no_budget_factoid_tier0():
    assert determine_tier("AUC 是多少") == Tier.WORKFLOW


def test_no_budget_analytical_tier1():
    assert determine_tier("帮我分析流失原因并提出召回建议") == Tier.EXPERT


# ── 自定义阈值 ────────────────────────────────────────────────────────────


def test_custom_tier0_threshold():
    assert determine_tier("分析", budget=800.0, tier0_budget_threshold=1000.0) == Tier.WORKFLOW


def test_custom_tier2_thresholds():
    assert (
        determine_tier("补贴留存广告全局优化", budget=8000.0, tier2_budget_threshold=10000.0)
        != Tier.DEEP
    )


# ── tier_degradation_chain ────────────────────────────────────────────────


def test_degradation_chain_from_deep():
    chain = tier_degradation_chain(Tier.DEEP)
    assert chain == [Tier.DEEP, Tier.EXPERT, Tier.WORKFLOW]


def test_degradation_chain_from_workflow():
    assert tier_degradation_chain(Tier.WORKFLOW) == [Tier.WORKFLOW]


# ── Tier 枚举 ─────────────────────────────────────────────────────────────


def test_tier_labels():
    assert Tier.WORKFLOW.label == "workflow"
    assert Tier.EXPERT.label == "expert"
    assert Tier.DEEP.label == "deep"


def test_tier_ordering():
    assert Tier.WORKFLOW < Tier.EXPERT < Tier.DEEP
