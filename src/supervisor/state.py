"""Supervisor 层的 state schema——在现有 AnalysisState 上扩展 tier 字段。

设计原则：不修改现有 AnalysisState（保持 v9 兼容），Supervisor 写入的
tier 字段作为可选扩展——旧消费者不读这些字段就不受影响。
"""

from __future__ import annotations

from typing import Any, TypedDict


class SupervisorState(TypedDict, total=False):
    """Supervisor 节点写入的 state 扩展字段。"""

    tier: int  # 0=workflow / 1=expert / 2=deep
    tier_label: str  # "workflow" / "expert" / "deep"
    tier_source: str  # "rule" / "llm" / "forced"
    tier_reason: str  # 判定依据（一句话）
    tier_domains: list[str]  # 命中的领域列表
    degradations: list[str]  # 降级标记（追加语义）


def merge_supervisor_state(base: dict[str, Any], sup: dict[str, Any]) -> dict[str, Any]:
    """将 Supervisor 输出合并进现有 state（追加 degradations，不覆盖其他）。"""
    out = {**base, **{k: v for k, v in sup.items() if k != "degradations"}}
    if sup.get("degradations"):
        existing = list(base.get("degradations") or [])
        out["degradations"] = existing + list(sup["degradations"])
    return out
