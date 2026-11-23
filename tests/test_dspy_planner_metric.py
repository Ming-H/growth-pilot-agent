"""dspy_planner 评分层单测（纯函数，无 dspy/LLM，CI 安全）。

覆盖：专家集合评分的分级语义、生产校验器清洗（白名单/去重/封顶/围栏剥离）、
honesty 空真值、规则基线可调用。
"""

from __future__ import annotations

from experiments.dspy_planner import metric as pm
from src.evals.tasks import Anchor, GrowthTask


def _task(experts: frozenset[str]) -> GrowthTask:
    return GrowthTask(
        id="t",
        query="q",
        budget=100.0,
        expected_experts=experts,
        anchors=(Anchor(label="a", value=1.0),),
    )


# ── score_experts 分级语义 ────────────────────────────────────────────────


def test_score_exact_match():
    assert pm.score_experts(frozenset({"subsidy"}), {"subsidy"}) == 1.0


def test_score_superset_full_coverage():
    # 全覆盖 + 多选：路由对但浪费预算
    assert pm.score_experts(frozenset({"subsidy"}), {"subsidy", "ad"}) == 0.75


def test_score_partial_coverage():
    # 双专家题只命中一个：0.5 × 0.5 召回
    assert pm.score_experts(frozenset({"retention", "subsidy"}), {"retention"}) == 0.25


def test_score_total_miss():
    assert pm.score_experts(frozenset({"prospect"}), {"ad"}) == 0.0


def test_score_honesty_empty_truth():
    # 真值为空（拒答题）：空规划 1.0，任何非空 0.0
    assert pm.score_experts(frozenset(), set()) == 1.0
    assert pm.score_experts(frozenset(), {"prospect"}) == 0.0


# ── parse_subgoals：生产校验器同一道闸 ────────────────────────────────────


def test_parse_plain_json():
    text = '{"subgoals": [{"expert": "subsidy", "question": "补贴怎么发"}]}'
    subs = pm.parse_subgoals(text, "q", 500.0)
    assert [s["expert"] for s in subs] == ["subsidy"]
    assert subs[0]["budget"] == 500.0  # subsidy 子目标继承会话预算


def test_parse_fenced_json():
    text = '```json\n{"subgoals": [{"expert": "prospect", "question": "高潜用户"}]}\n```'
    subs = pm.parse_subgoals(text, "q")
    assert [s["expert"] for s in subs] == ["prospect"]


def test_parse_drops_whitelist_violation_and_dupe():
    text = (
        '{"subgoals": ['
        '{"expert": "subsidy", "question": "a"},'
        '{"expert": "finance", "question": "b"},'
        '{"expert": "subsidy", "question": "c"}]}'
    )
    subs = pm.parse_subgoals(text, "q")
    assert [s["expert"] for s in subs] == ["subsidy"]  # 白名单外丢弃 + 去重


def test_parse_garbage_returns_empty():
    assert pm.parse_subgoals("模型自由发挥", "q") == []
    assert pm.parse_subgoals('{"subgoals": "not-a-list"}', "q") == []


# ── score_task：明细结构 ──────────────────────────────────────────────────


def test_score_task_fields():
    r = pm.score_task(_task(frozenset({"subsidy"})), [{"expert": "subsidy", "question": "x"}])
    assert r["score"] == 1.0 and r["expected"] == ["subsidy"] and r["got"] == ["subsidy"]


def test_honesty_task_requires_empty_plan():
    r = pm.score_task(_task(frozenset()), [])
    assert r["score"] == 1.0
    r2 = pm.score_task(_task(frozenset()), [{"expert": "ad", "question": "x"}])
    assert r2["score"] == 0.0


# ── 规则基线可调用（不评分数值，只验证通道） ─────────────────────────────


def test_rule_baseline_returns_whitelisted_experts():
    got = pm.rule_baseline_experts("分析本月用户增长，看补贴 ROI")
    assert isinstance(got, set)
    assert got <= {"prospect", "conversion", "subsidy", "retention", "ad", "budget"}
