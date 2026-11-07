"""BudgetExpert - global cross-channel budget orchestration.

The coordinating layer the five domain experts lacked: given ONE budget,
decide how much goes to 投放 (ad) / 补贴 (subsidy) / 召回 (retention) /
券 (conversion). Deterministic response curves (optionally calibrated by
KPIs sibling experts already produced in this run) + integer
programming. The domain experts answer "how does my channel work";
this expert answers "how does the budget flow between them".
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.expert import ExpertAgentBase
from src.prompts.templates.agent_prompts import BudgetPrompt

logger = logging.getLogger(__name__)

try:
    from src.tools.budget import ChannelResponseModel, GlobalBudgetAllocator
except (ImportError, OSError) as _import_err:
    import logging as _logging

    _logging.getLogger(__name__).warning("Tool import failed, using stubs: %s", _import_err)

    class _Stub:
        """Stub that raises RuntimeError when any method is called."""

        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            def _stub_method(*a: Any, **kw: Any) -> Any:
                raise RuntimeError(
                    f"Stub tool: {name}() called but tool is not available (import failed)"
                )

            return _stub_method

    ChannelResponseModel = GlobalBudgetAllocator = _Stub  # type: ignore[assignment,misc]


class BudgetExpert(ExpertAgentBase):
    """Global budget orchestration across the four spend channels."""

    name = "budget"
    description = "预算统筹专家 Agent"

    # ------------------------------------------------------------------
    # ExpertAgentBase interface
    # ------------------------------------------------------------------

    def _init_tools(self) -> dict[str, Any]:
        return {
            "response_model": ChannelResponseModel,
            "allocator": GlobalBudgetAllocator(),
        }

    def _get_system_prompt(self) -> str:
        template = BudgetPrompt()
        return f"{template.role_definition}\n\n{template.business_context}"

    @staticmethod
    def can_handle(query: str) -> float:
        """Global-allocation semantics only — never poach plain '预算' (subsidy owns it)."""
        return ExpertAgentBase._keyword_confidence(
            query,
            [
                "跨渠道",
                "统筹",
                "全局预算",
                "预算分配",
                "预算切分",
                "预算划分",
                "渠道间",
                "预算怎么分",
                "之间怎么分",
                "budget allocation",
                "allocate budget",
                "marketing mix",
            ],
        )

    # ------------------------------------------------------------------
    # Main pipeline (deterministic)
    # ------------------------------------------------------------------

    def _execute_pipeline(self, params: dict) -> dict:
        budget = float(params.get("budget") or 0)
        if budget <= 0:
            # analyze() turns the exception into success=False + errors —
            # the honest failure contract shared by every expert.
            raise ValueError("budget 参数缺失或非正数（全局预算统筹必须提供 budget，单位 CNY）")

        context_kpis = params.get("context_kpis") or {}
        model = ChannelResponseModel(total_budget=budget, context_kpis=context_kpis)
        grids = {c: model.grid(c) for c in model.channels}
        plan = self._tools["allocator"].allocate(grids, total_budget=budget)
        if not isinstance(plan, dict) or "error" in plan:
            raise ValueError(f"allocator failed: {(plan or {}).get('error', 'unknown')}")

        calibrated = any(model.multiplier(c) != 1.0 for c in model.channels)
        allocation = plan["allocation"]
        kpis: dict[str, float] = {
            "budget_total": float(budget),
            "expected_incremental_gmv": float(plan["expected_incremental_gmv"]),
            "expected_roi": float(plan["expected_roi"]),
        }
        for channel, spend in allocation.items():
            kpis[f"alloc_{channel}"] = float(spend)

        return {
            "allocation": allocation,
            "channel_details": plan["channel_details"],
            "total_spend": plan["total_spend"],
            "unallocated": plan["unallocated"],
            "expected_incremental_gmv": plan["expected_incremental_gmv"],
            "expected_roi": plan["expected_roi"],
            "solver_method": plan["solver_method"],
            "data_source": "calibrated" if calibrated else "default_curves",
            "calibration_multipliers": {c: model.multiplier(c) for c in model.channels},
            "kpis": kpis,
        }

    # ------------------------------------------------------------------
    # LLM synthesis prompt
    # ------------------------------------------------------------------

    def _build_synthesis_prompt(self, params: dict, results: dict) -> str:
        rows = "\n".join(
            f"- {c}: spend ¥{d['spend']:,.0f} → 预期增量GMV ¥{d['expected_incremental_gmv']:,.0f}"
            f"（ROI {d['expected_roi']}，占比 {d['budget_share']:.0%}）"
            for c, d in results.get("channel_details", {}).items()
        )
        return (
            f"用户问题: {params.get('query', '')}\n"
            f"总预算: ¥{float(params.get('budget') or 0):,.0f}\n\n"
            f"分配方案（求解器: {results.get('solver_method')}，"
            f"数据口径: {results.get('data_source')}）:\n{rows}\n\n"
            "请解读该分配方案：为什么边际收益曲线下这是最优切分、各渠道的职责、"
            "风险与下一步实验建议。所有数字必须来自上述结果。"
        )
