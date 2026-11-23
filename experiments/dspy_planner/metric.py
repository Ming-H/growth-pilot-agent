"""dspy_planner 评测层（纯函数，无 dspy/LLM 依赖，CI 可测）。

被优化的对象是 agent 模式的规划节点（``src/agent/chief.py`` 的
``agent_plan_node``）：它把用户问题拆解为带专家指派的子目标清单，
质量直接决定路由正确性。本模块提供三件事：

1. ``parse_subgoals`` —— 候选输出经**生产校验器**（``_validate_subgoals``，
   只删不造）清洗，与线上同一道闸；
2. ``score_experts`` —— 候选专家集合 vs 任务真值（``builtin_tasks`` 的
   ``expected_experts``）的评分：精确匹配 1.0，超集（全覆盖+多选）0.75，
   部分覆盖按召回减半，honesty 任务（真值为空）要求空规划；
3. ``rule_baseline_experts`` —— 生产规则路由（``_intent_classify``，
   ``plan:rule`` 兜底同款）作为零成本基线。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.agent.chief import _validate_subgoals  # noqa: E402


def parse_subgoals(text: str, query: str, budget: float | None = None) -> list[dict[str, Any]]:
    """候选文本 → 合法子目标清单（复用生产校验器，含 ``` 围栏剥离）。"""
    content = (text or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return []
    return _validate_subgoals(raw, query, budget)


def score_experts(expected: frozenset[str], got: set[str]) -> float:
    """专家集合评分：真值为空（honesty）⇔ 候选也须为空。

    精确 1.0；超集 0.75（全覆盖但多选，浪费预算）；部分覆盖 0.5×召回；
    全漏 0。
    """
    if not expected:
        return 1.0 if not got else 0.0
    if not got:
        return 0.0
    if expected == got:
        return 1.0
    inter = expected & got
    recall = len(inter) / len(expected)
    if recall == 1.0:
        return 0.75
    return round(0.5 * recall, 3)


def rule_baseline_experts(query: str) -> set[str]:
    """生产规则路由（``plan:rule`` 兜底同款）选出的专家集合。"""
    from src.graph.nodes import _intent_classify

    return set(_intent_classify(query))


def score_task(task, subgoals: list[dict[str, Any]]) -> dict[str, Any]:
    """单任务评分：候选子目标（已经 parse_subgoals 清洗）→ 分数与明细。"""
    got = {sg["expert"] for sg in subgoals}
    return {
        "id": task.id,
        "expected": sorted(task.expected_experts),
        "got": sorted(got),
        "score": score_experts(task.expected_experts, got),
    }
