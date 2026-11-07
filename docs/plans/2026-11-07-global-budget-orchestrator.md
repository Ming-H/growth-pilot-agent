# 全局预算统筹层（BudgetExpert）实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 新增第 6 个专家 `budget`（预算统筹专家）：在 投放/补贴/召回/券 四个渠道间做全局预算分配（凹响应曲线 + 整数规划），补上"跨专家全局资源配置层"这一结构性缺口。

**Architecture:** 做成 `ExpertAgentBase` 子类而非裸工具（对比 `search_memory`），从而白得：Pydantic 结果校验 + `_validated` 标记（→ L1 台账写回）、`_intent_classify` 统一路由、workflow 模式自动接入、hook 生命周期与 `threadpool_limits(1)` 确定性包装。响应曲线 `g(s) = scale·log1p(s/k)`（边际递减），可被同 run 内其他专家已产出的 KPI 有界校准（`context_kpis`，agent 模式由 tools 节点从 `kpi_snapshot` 注入）；分配问题用 PuLP+HiGHS 单选 IP 求解，无求解器时用精确枚举兜底（4 渠道×11 档 = 12⁴ 组合，秒级）。

**Tech Stack:** PuLP + HiGHS（复用现有依赖）、Pydantic v2、LangGraph（不动图结构）、pytest。

**关键设计约束（不可破坏）：**

1. **确定性**：默认曲线参数为常数；校准乘子有界 [0.5, 2.0]；PuLP 固定输入定长输出；零 key 模式两次运行 kpis 逐字节一致。
2. **路由不抢词**：subsidy 的 can_handle 含 `"预算"`、`"budget"`。budget 专家关键词必须是"全局/跨渠道"语义（跨渠道/统筹/预算分配/渠道间…），纯"补贴预算优化"不得命中 budget（Task 5 有回归测试）。
3. **KPI 闸门协同**：分配数值进 `kpis`（float），报告中"投放分得 72,000"可回溯；solver 名等字符串不进 kpis。
4. **只删不造**：planner 白名单加 `"budget"` 是扩集合，validator 行为不变。
5. **budget 缺失即失败**：`budget<=0` 时 pipeline 抛 `ValueError` → `analyze()` 记 errors → `success=False` → 不写 L1 台账（与其他专家的失败语义一致）。

**测试运行环境（所有 pytest 命令同此 env）：**

```bash
GPA_DEMO_MODE=true GPA_LLM_API_KEY=sk-test-placeholder GPA_DB_URL=sqlite+aiosqlite:///./test.db \
uv run python -m pytest <目标> -v --tb=short
```

---

## Task 1: 渠道响应曲线模型 `ChannelResponseModel`

**Files:**
- Create: `src/tools/budget/__init__.py`
- Create: `src/tools/budget/response_curves.py`
- Test: `tests/test_tools/test_budget_tools.py`

**Step 1: 写失败测试**

```python
"""Tests for the global budget orchestration tools."""

from __future__ import annotations

import math

import pytest


class TestChannelResponseModel:
    """Concave response curves + bounded calibration."""

    def _make(self, budget: float = 200_000.0, context_kpis: dict | None = None):
        from src.tools.budget.response_curves import ChannelResponseModel

        return ChannelResponseModel(total_budget=budget, context_kpis=context_kpis or {})

    def test_channels_exposed(self):
        m = self._make()
        assert set(m.channels) == {"ad", "subsidy", "retention", "conversion"}

    def test_zero_spend_zero_gmv(self):
        m = self._make()
        grid = m.grid("ad")
        assert grid[0] == (0.0, 0.0)

    def test_curve_is_concave(self):
        """Marginal returns must be non-increasing (saturation)."""
        m = self._make()
        for channel in m.channels:
            grid = m.grid(channel)
            marginals = [grid[i + 1][1] - grid[i][1] for i in range(len(grid) - 1)]
            assert all(
                marginals[i + 1] <= marginals[i] + 1e-9 for i in range(len(marginals) - 1)
            ), f"{channel} marginal returns increased"

    def test_grid_respects_channel_cap(self):
        """Max grid spend ≤ min(max_share * budget, max_abs)."""
        m = self._make(budget=1_000_000.0)
        for channel in m.channels:
            cap = m.channel_cap(channel)
            assert grid_max := max(s for s, _ in m.grid(channel)) <= cap + 1e-6

    def test_calibration_multiplier_applied_and_bounded(self):
        """observed ROI 4x the reference → scale multiplier clamped at 2.0."""
        from src.tools.budget.response_curves import CALIBRATION_RULES

        ref = CALIBRATION_RULES["subsidy"]["reference_roi"]
        m = self._make(context_kpis={"subsidy_expected_roi": ref * 4.0})
        base = self._make()
        s_cal = m.grid("subsidy")[-1][1]
        s_base = base.grid("subsidy")[-1][1]
        assert s_cal == pytest.approx(s_base * 2.0, rel=1e-6)

    def test_no_calibration_keys_multiplier_one(self):
        m = self._make(context_kpis={"unrelated": 3.3})
        base = self._make()
        assert m.grid("ad") == base.grid("ad")

    def test_deterministic_same_input_same_output(self):
        a = self._make(context_kpis={"subsidy_expected_roi": 2.0}).grid("subsidy")
        b = self._make(context_kpis={"subsidy_expected_roi": 2.0}).grid("subsidy")
        assert a == b
```

**Step 2: 运行确认失败**

```bash
GPA_DEMO_MODE=true GPA_LLM_API_KEY=sk-test-placeholder GPA_DB_URL=sqlite+aiosqlite:///./test.db \
uv run python -m pytest tests/test_tools/test_budget_tools.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.tools.budget'`

**Step 3: 实现**

`src/tools/budget/__init__.py`:
```python
"""Global budget orchestration tools (response curves + allocator)."""

from src.tools.budget.response_curves import ChannelResponseModel
from src.tools.budget.global_allocator import GlobalBudgetAllocator

__all__ = ["ChannelResponseModel", "GlobalBudgetAllocator"]
```

`src/tools/budget/response_curves.py`:
```python
"""Channel response models for global budget orchestration.

Each growth channel is modelled with a concave (saturating) response
curve: incremental GMV as a function of spend, ``g(s) = scale * log1p(s / k)``.
Marginal returns decline — the property that makes cross-channel
allocation a real optimisation instead of "give everything to the best
channel".

Determinism: default parameters are constants; calibration from
``context_kpis`` (KPIs already produced by other experts in the same
run) applies a bounded multiplier to ``scale``. Same inputs → same
curve, always.
"""

from __future__ import annotations

import math
from typing import Any

# Channel defaults. scale/k shape the log1p saturation curve; max_share
# caps the share of the total budget a single channel may absorb;
# max_abs is an absolute ceiling (so a huge budget cannot push 60% of
# ¥10M into retention). Values are plausible-growth defaults — the
# honest label is data_source=default_curves until calibrated.
CHANNEL_DEFAULTS: dict[str, dict[str, float]] = {
    "ad": {"scale": 420_000.0, "k": 150_000.0, "max_share": 0.60, "max_abs": 600_000.0},
    "subsidy": {"scale": 380_000.0, "k": 90_000.0, "max_share": 0.60, "max_abs": 400_000.0},
    "retention": {"scale": 150_000.0, "k": 40_000.0, "max_share": 0.40, "max_abs": 150_000.0},
    "conversion": {"scale": 200_000.0, "k": 60_000.0, "max_share": 0.40, "max_abs": 200_000.0},
}

# First-order calibration: anchor each channel's scale to an ROI KPI a
# sibling expert may already have produced in this run. The multiplier
# is clamped — calibration is anchoring, not curve re-fitting.
CALIBRATION_RULES: dict[str, dict[str, Any]] = {
    "ad": {"kpi_keys": ["ad_expected_roi", "ad_roi"], "reference_roi": 2.0},
    "subsidy": {"kpi_keys": ["subsidy_expected_roi", "expected_roi"], "reference_roi": 1.6},
    "retention": {"kpi_keys": ["retention_expected_roi", "winback_roi"], "reference_roi": 1.8},
    "conversion": {"kpi_keys": ["conversion_expected_roi", "coupon_roi"], "reference_roi": 2.2},
}

_CALIBRATION_BOUNDS = (0.5, 2.0)
_GRID_STEPS = 10  # 0%, 10%, ..., 100% of the channel cap


def _as_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class ChannelResponseModel:
    """Saturating response curves per channel, optionally calibrated."""

    def __init__(self, total_budget: float, context_kpis: dict[str, Any] | None = None) -> None:
        if total_budget <= 0:
            raise ValueError("total_budget must be positive")
        self.total_budget = float(total_budget)
        self.context_kpis = context_kpis or {}
        self._multipliers = {
            channel: self._calibration_multiplier(channel) for channel in CHANNEL_DEFAULTS
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def channels(self) -> list[str]:
        return list(CHANNEL_DEFAULTS)

    def channel_cap(self, channel: str) -> float:
        d = CHANNEL_DEFAULTS[channel]
        return round(min(d["max_share"] * self.total_budget, d["max_abs"]), 2)

    def curve(self, channel: str) -> tuple[float, float]:
        """Return the (scale, k) parameters after calibration."""
        d = CHANNEL_DEFAULTS[channel]
        return d["scale"] * self._multipliers[channel], d["k"]

    def grid(self, channel: str) -> list[tuple[float, float]]:
        """(spend, incremental_gmv) grid from 0 to the channel cap."""
        cap = self.channel_cap(channel)
        scale, k = self.curve(channel)

        def g(s: float) -> float:
            return round(scale * math.log1p(s / k), 2) if s > 0 else 0.0

        return [(round(cap * i / _GRID_STEPS, 2), g(cap * i / _GRID_STEPS)) for i in range(_GRID_STEPS + 1)]

    def multiplier(self, channel: str) -> float:
        """The applied calibration multiplier (1.0 = default curve)."""
        return self._multipliers[channel]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _calibration_multiplier(self, channel: str) -> float:
        rule = CALIBRATION_RULES[channel]
        observed = None
        for key in rule["kpi_keys"]:
            v = _as_float(self.context_kpis.get(key))
            if v is not None and v > 0:
                observed = v
                break
        if observed is None:
            return 1.0
        ratio = observed / float(rule["reference_roi"])
        return round(min(max(ratio, _CALIBRATION_BOUNDS[0]), _CALIBRATION_BOUNDS[1]), 4)
```

注意：`__init__.py` 里 `GlobalBudgetAllocator` 的 import 在 Task 2 才存在——Task 1 先不写该行（或写 `from src.tools.budget.global_allocator import ...` 会 ImportError）。**Task 1 的 `__init__.py` 只导出 `ChannelResponseModel`，Task 2 再补第二行。**

**Step 4: 跑测试确认通过**

Expected: `7 passed`

**Step 5: Commit**

```bash
git add src/tools/budget/ tests/test_tools/test_budget_tools.py
git commit -m "feat(budget): concave channel response curves with bounded KPI calibration"
```

---

## Task 2: 全局分配器 `GlobalBudgetAllocator`

**Files:**
- Create: `src/tools/budget/global_allocator.py`
- Modify: `src/tools/budget/__init__.py`（补第二行 export）
- Test: `tests/test_tools/test_budget_tools.py`（追加）

**Step 1: 追加失败测试**

```python
class TestGlobalBudgetAllocator:
    """IP allocation with exact-enumeration fallback."""

    def _grids(self, budget: float = 200_000.0):
        from src.tools.budget.response_curves import ChannelResponseModel

        m = ChannelResponseModel(total_budget=budget)
        return {c: m.grid(c) for c in m.channels}

    def _alloc(self, budget: float = 200_000.0, **kw):
        from src.tools.budget.global_allocator import GlobalBudgetAllocator

        return GlobalBudgetAllocator().allocate(self._grids(budget), total_budget=budget, **kw)

    def test_allocation_respects_total_budget(self):
        r = self._alloc()
        assert r["status"] == "optimal"
        assert sum(r["allocation"].values()) <= 200_000.0 + 1e-6

    def test_best_channel_gets_largest_share(self):
        """Subsidy has the strongest low-spend response in the defaults."""
        r = self._alloc()
        assert max(r["allocation"], key=r["allocation"].get) in {"subsidy", "ad"}

    def test_per_channel_cap_respected(self):
        from src.tools.budget.response_curves import ChannelResponseModel

        m = ChannelResponseModel(total_budget=200_000.0)
        r = self._alloc()
        for channel, spend in r["allocation"].items():
            assert spend <= m.channel_cap(channel) + 1e-6

    def test_zero_budget_errors(self):
        r = self._alloc(budget=0)
        assert "error" in r

    def test_expected_roi_consistent(self):
        r = self._alloc()
        spent = sum(r["allocation"].values())
        assert r["expected_incremental_gmv"] >= 0
        if spent > 0:
            assert r["expected_roi"] == pytest.approx(
                r["expected_incremental_gmv"] / spent, rel=1e-6
            )

    def test_enumeration_fallback_matches_ip(self):
        """With PuLP monkeypatched out, enumeration must find the same optimum."""
        import src.tools.budget.global_allocator as ga
        from src.tools.budget.response_curves import ChannelResponseModel

        grids = self._grids()
        ip = ga.GlobalBudgetAllocator().allocate(grids, total_budget=200_000.0)
        real_has = ga.HAS_PULP
        ga.HAS_PULP = False
        try:
            enum = ga.GlobalBudgetAllocator().allocate(grids, total_budget=200_000.0)
        finally:
            ga.HAS_PULP = real_has
        assert enum["solver_method"] == "enumeration"
        assert enum["expected_incremental_gmv"] == pytest.approx(
            ip["expected_incremental_gmv"], rel=1e-9
        )

    def test_deterministic(self):
        a = self._alloc()
        b = self._alloc()
        assert a["allocation"] == b["allocation"]
        assert a["expected_incremental_gmv"] == b["expected_incremental_gmv"]
```

**Step 2: 运行确认失败** — `ModuleNotFoundError: No module named 'src.tools.budget.global_allocator'`

**Step 3: 实现**

`src/tools/budget/global_allocator.py`:
```python
"""GlobalBudgetAllocator - cross-channel budget allocation.

Maximise total expected incremental GMV subject to the total budget and
per-channel caps, over the discrete (spend, gmv) grids produced by
``ChannelResponseModel``. Primary solver: PuLP + HiGHS MIP (single-
selection formulation). Fallback: exhaustive enumeration — exact at this
scale (4 channels x 11 levels = 12^4 combinations) and dependency-free,
so the allocator degrades in *method*, never in *correctness*.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pulp

    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    logger.warning("PuLP not available; global allocation falls back to enumeration")


class GlobalBudgetAllocator:
    """Allocate one budget across channel grids (concave curves)."""

    def allocate(
        self,
        channel_grids: dict[str, list[tuple[float, float]]],
        total_budget: float,
    ) -> dict[str, Any]:
        if total_budget <= 0:
            return {"error": "total_budget must be positive"}
        if not channel_grids:
            return {"error": "channel_grids is empty"}

        if HAS_PULP:
            try:
                result = self._solve_ip(channel_grids, total_budget)
                result["solver_method"] = "ip-highs"
                return result
            except Exception as exc:  # solver trouble must never break the pipeline
                logger.warning("[global_allocator] IP failed (%s); enumerating", exc)

        result = self._enumerate(channel_grids, total_budget)
        result["solver_method"] = "enumeration"
        return result

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _solve_ip(
        self, channel_grids: dict[str, list[tuple[float, float]]], total_budget: float
    ) -> dict[str, Any]:
        """Single-selection MIP: each channel picks at most one grid level."""
        prob = pulp.LpProblem("global_budget", pulp.LpMaximize)
        x: dict[tuple[str, int], pulp.LpVariable] = {}
        for channel, grid in channel_grids.items():
            for level in range(len(grid)):
                x[(channel, level)] = pulp.LpVariable(
                    f"x_{channel}_{level}", cat="Binary"
                )

        prob += pulp.lpSum(
            grid[level][1] * x[(channel, level)]
            for channel, grid in channel_grids.items()
            for level in range(len(grid))
        )
        # at most one level per channel
        for channel, grid in channel_grids.items():
            prob += pulp.lpSum(x[(channel, l)] for l in range(len(grid))) <= 1
        # total spend within budget
        prob += (
            pulp.lpSum(
                grid[level][0] * x[(channel, level)]
                for channel, grid in channel_grids.items()
                for level in range(len(grid))
            )
            <= total_budget
        )

        prob.solve(pulp.HiGHS(msg=False))  # type: ignore[attr-defined]
        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError(f"solver status {pulp.LpStatus[prob.status]}")

        allocation: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}
        for channel, grid in channel_grids.items():
            chosen = next(
                (l for l in range(len(grid)) if x[(channel, l)].value() == 1), 0
            )
            spend, gmv = grid[chosen]
            allocation[channel] = spend
            details[channel] = {
                "spend": spend,
                "expected_incremental_gmv": gmv,
                "expected_roi": round(gmv / spend, 4) if spend > 0 else 0.0,
                "budget_share": round(spend / total_budget, 4),
            }
        return self._assemble(allocation, details, total_budget)

    def _enumerate(
        self, channel_grids: dict[str, list[tuple[float, float]]], total_budget: float
    ) -> dict[str, Any]:
        """Exact fallback: iterate every combination (deterministic order)."""
        channels = list(channel_grids)
        best: tuple[float, dict[str, int]] = (0.0, {c: 0 for c in channels})
        for combo in itertools.product(*(range(len(channel_grids[c])) for c in channels)):
            spend = sum(channel_grids[c][l][0] for c, l in zip(channels, combo))
            if spend > total_budget + 1e-9:
                continue
            value = sum(channel_grids[c][l][1] for c, l in zip(channels, combo))
            if value > best[0] + 1e-9:  # strict: first-best wins ties
                best = (value, dict(zip(channels, combo)))

        allocation = {c: channel_grids[c][l][0] for c, l in zip(channels, best[1])}
        details = {
            c: {
                "spend": channel_grids[c][l][0],
                "expected_incremental_gmv": channel_grids[c][l][1],
                "expected_roi": (
                    round(channel_grids[c][l][1] / channel_grids[c][l][0], 4)
                    if channel_grids[c][l][0] > 0
                    else 0.0
                ),
                "budget_share": round(channel_grids[c][l][0] / total_budget, 4),
            }
            for c, l in zip(channels, best[1])
        }
        return self._assemble(allocation, details, total_budget)

    @staticmethod
    def _assemble(
        allocation: dict[str, float], details: dict[str, dict[str, Any]], total_budget: float
    ) -> dict[str, Any]:
        total_gmv = round(sum(d["expected_incremental_gmv"] for d in details.values()), 2)
        spent = round(sum(allocation.values()), 2)
        return {
            "status": "optimal",
            "allocation": allocation,
            "channel_details": details,
            "total_spend": spent,
            "unallocated": round(total_budget - spent, 2),
            "expected_incremental_gmv": total_gmv,
            "expected_roi": round(total_gmv / spent, 4) if spent > 0 else 0.0,
        }
```

同时更新 `src/tools/budget/__init__.py` 补上：
```python
from src.tools.budget.global_allocator import GlobalBudgetAllocator
```

**Step 4: 跑测试确认通过** — `15 passed`（Task1 7 + Task2 8）

**Step 5: Commit**

```bash
git add src/tools/budget/ tests/test_tools/test_budget_tools.py
git commit -m "feat(budget): global cross-channel allocator (IP-HiGHS + exact enumeration fallback)"
```

---

## Task 3: 结果模型 `BudgetResult` + 注册

**Files:**
- Modify: `src/core/models.py`（AdResult 之后加类；`EXPERT_RESULT_MODELS` 加映射）
- Test: `tests/test_tools/test_budget_tools.py`（追加，经 `_validate_expert_result` 验证）

**Step 1: 追加失败测试**

```python
class TestBudgetResultModel:
    def test_validate_expert_result_marks_validated(self):
        from src.core.models import EXPERT_RESULT_MODELS
        from src.graph.nodes import _validate_expert_result

        assert "budget" in EXPERT_RESULT_MODELS
        result = {
            "success": True,
            "allocation": {"ad": 60_000.0, "subsidy": 90_000.0},
            "expected_incremental_gmv": 320_000.0,
            "expected_roi": 1.8,
            "kpis": {"budget_total": 200_000.0, "expected_roi": 1.8},
        }
        out = _validate_expert_result("budget", result)
        assert out["_validated"] is True
        assert out["agent_name"] == "budget"
```

**Step 2: 运行确认失败** — `assert "budget" in EXPERT_RESULT_MODELS` AssertionError

**Step 3: 实现**（`src/core/models.py`，仿 `AdResult`，放在其类定义之后）

```python
class BudgetResult(AgentResult):
    """Result from BudgetExpert (global budget orchestration)."""

    agent_name: str = "budget"
    allocation: dict[str, float] = Field(default_factory=dict)
    channel_details: dict[str, Any] = Field(default_factory=dict)
    expected_incremental_gmv: float = 0.0
    expected_roi: float = 0.0
    total_spend: float = 0.0
    unallocated: float = 0.0
    solver_method: str = ""
    data_source: str = "default_curves"
    analysis: LLMAnalysis = Field(default_factory=LLMAnalysis)
```

`EXPERT_RESULT_MODELS` 增加一行 `"budget": BudgetResult,`。

**Step 4: 确认通过** — `16 passed`

**Step 5: Commit** — `git commit -m "feat(budget): BudgetResult pydantic model + expert registry entry"`

---

## Task 4: `BudgetExpert` 专家 + Prompt 模板

**Files:**
- Create: `src/agents/budget.py`
- Modify: `src/agents/__init__.py`（注册导出）
- Modify: `src/prompts/templates/agent_prompts.py`（加 `BudgetPrompt`，**先读 RetentionPrompt 类的完整形状——role_definition/business_context 的定义位置——严格同构**）
- Test: `tests/test_tools/test_budget_tools.py`（追加）

**Step 1: 追加失败测试**

```python
class TestBudgetExpert:
    """Pipeline integration: params → allocation + KPIs, zero-key deterministic."""

    def _run(self, budget: float = 200_000.0, context_kpis: dict | None = None):
        from src.agents.budget import BudgetExpert

        return BudgetExpert(llm=None).analyze(
            {"budget": budget, "query": "全局预算分配", "context_kpis": context_kpis or {}}
        )

    def test_pipeline_success_and_kpis(self):
        r = self._run()
        assert r["success"] is True
        assert set(r["allocation"]) == {"ad", "subsidy", "retention", "conversion"}
        assert r["kpis"]["budget_total"] == 200_000.0
        assert r["kpis"]["expected_roi"] > 0
        # gate-verifiable numbers only (floats) in kpis
        assert all(isinstance(v, float) for v in r["kpis"].values())

    def test_missing_budget_fails_honestly(self):
        r = BudgetExpert_and_run = self._run(budget=0)
        assert r["success"] is False
        assert r["errors"]

    def test_context_kpis_change_allocation(self):
        """Calibrated curves (subsidy ROI x2) must shift allocation to subsidy."""
        base = self._run()
        cal = self._run(context_kpis={"subsidy_expected_roi": 3.2})
        assert cal["data_source"] == "calibrated"
        assert cal["allocation"]["subsidy"] >= base["allocation"]["subsidy"]

    def test_can_handle_keywords(self):
        from src.agents.budget import BudgetExpert

        assert BudgetExpert.can_handle("跨渠道预算怎么分配") >= 0.5
        assert BudgetExpert.can_handle("预算在投放和补贴之间怎么分") >= 0.5
        # must NOT poach plain subsidy-budget queries
        assert BudgetExpert.can_handle("补贴预算优化") < 0.5
        assert BudgetExpert.can_handle("优惠券设计") == 0.0

    def test_zero_key_byte_deterministic(self):
        a, b = self._run(), self._run()
        assert a["kpis"] == b["kpis"]
        assert a["allocation"] == b["allocation"]
```

**Step 2: 确认失败** — `ModuleNotFoundError: src.agents.budget`

**Step 3: 实现**（先读 `agent_prompts.py` 中 RetentionPrompt 全文照抄结构）

`src/agents/budget.py`:
```python
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
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        def __getattr__(self, name: str) -> Any:
            def _stub_method(*a: Any, **kw: Any) -> Any:
                raise RuntimeError(f"Stub tool: {name}() unavailable (import failed)")

            return _stub_method

    ChannelResponseModel = GlobalBudgetAllocator = _Stub  # type: ignore[assignment,misc]


class BudgetExpert(ExpertAgentBase):
    """Global budget orchestration across the four spend channels."""

    name = "budget"
    description = "预算统筹专家 Agent"

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

    def _execute_pipeline(self, params: dict) -> dict:
        budget = float(params.get("budget") or 0)
        if budget <= 0:
            # analyze() turns the exception into success=False + errors —
            # the honest failure contract shared by every expert.
            raise ValueError("budget 参数缺失或非正数（全局预算统筹必须提供 budget, 单位 CNY）")

        context_kpis = params.get("context_kpis") or {}
        model = ChannelResponseModel(total_budget=budget, context_kpis=context_kpis)
        grids = {c: model.grid(c) for c in model.channels}
        plan = self._safe_execute(
            lambda: self._tools["allocator"].allocate(grids, total_budget=budget),
            "GlobalBudgetAllocator",
            [],  # collected by analyze(); pass-through
            default=None,
        )
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

    def _build_synthesis_prompt(self, params: dict, results: dict) -> str:
        rows = "\n".join(
            f"- {c}: spend ¥{d['spend']:,.0f} → 预期增量GMV ¥{d['expected_incremental_gmv']:,.0f}"
            f" (ROI {d['expected_roi']}, 占比 {d['budget_share']:.0%})"
            for c, d in results.get("channel_details", {}).items()
        )
        return (
            f"用户问题: {params.get('query', '')}\n总预算: ¥{params.get('budget', 0):,.0f}\n\n"
            f"分配方案（求解器: {results.get('solver_method')}, "
            f"数据口径: {results.get('data_source')}）:\n{rows}\n\n"
            "请解读该分配方案：为什么边际收益曲线下这是最优切分、各渠道的职责、"
            "风险与下一步实验建议。所有数字必须来自上述结果。"
        )
```

`BudgetPrompt`（结构完全照 RetentionPrompt 同构，角色定义按预算统筹改写；DEFAULT_CONTEXT 含 user_query/budget/allocation_summary/channel_details/solver_method/data_source）。

`src/agents/__init__.py` 加 import + `__all__`。

**Step 4: 确认通过** — `21 passed`

**Step 5: Commit** — `git commit -m "feat(budget): BudgetExpert — global cross-channel orchestration expert (zero-key deterministic)"`

---

## Task 5: 图层接线（路由/工厂/白名单/规划提示词）

**Files:**
- Modify: `src/graph/nodes.py` — ① `_intent_classify` 的专家列表加 `("budget", "src.agents.budget.BudgetExpert")`；② `_get_expert_cls` 的 `import_map` 加同键；③ `PLAN_PROMPT` 专家清单加一行 `- budget (预算统筹): cross-channel budget allocation across ad/subsidy/retention/conversion`
- Modify: `src/agent/chief.py:34` — `_EXPERT_WHITELIST` 加 `"budget"`
- Test: `tests/test_graph/test_nodes.py`（追加）

**Step 1: 追加失败测试**

```python
class TestBudgetRouting:
    def test_intent_classify_routes_global_budget_query(self):
        from src.graph.nodes import _intent_classify

        selected = _intent_classify("预算在投放和补贴之间怎么分配？")
        assert "budget" in selected

    def test_intent_classify_does_not_poach_subsidy_budget(self):
        from src.graph.nodes import _intent_classify

        selected = _intent_classify("补贴预算优化怎么做")
        assert "budget" not in selected
        assert "subsidy" in selected

    def test_create_expert_budget(self):
        from src.agents.budget import BudgetExpert
        from src.graph.nodes import _create_expert

        assert isinstance(_create_expert("budget"), BudgetExpert)

    def test_chief_whitelist_includes_budget(self):
        from src.agent.chief import _EXPERT_WHITELIST

        assert "budget" in _EXPERT_WHITELIST
```

**Step 2: 确认失败** — `assert "budget" in selected` AssertionError

**Step 3: 实现** — 上述 4 处一行级修改。

**Step 4: 确认通过**（含 test_nodes 既有测试不回归）

**Step 5: Commit** — `git commit -m "feat(budget): wire budget expert into router, expert factory, planner whitelist"`

---

## Task 6: Chief Agent 工具注册 + context_kpis 注入

**Files:**
- Modify: `src/agent/tools.py` — ① `EXPERT_TOOL_DESCRIPTIONS` 加 `budget`（ACI 四段式）；② `EXPERT_TOOL_FAILURE_MODES` 加 `["missing_budget", "default_curves_uncalibrated", "solver_fallback_enumeration"]`；③ `_run_expert_tool` 的 `params` 增加 `"context_kpis": dict(state.get("kpi_snapshot") or {})`（所有专家统一注入，budget 消费，其他专家忽略——同 run 内已产出 KPI 成为全局分配的校准锚点）
- Test: `tests/test_agent/test_tools.py`（追加）

**Step 1: 追加失败测试**

```python
class TestBudgetToolSpec:
    def test_budget_tool_registered_with_failure_modes(self):
        from src.agent.tools import EXPERT_TOOL_FAILURE_MODES, get_spec_by_name

        spec = get_spec_by_name("run_budget_analysis")
        assert spec is not None
        assert "budget" in EXPERT_TOOL_FAILURE_MODES
        assert "跨渠道" in spec.description

    @pytest.mark.asyncio
    async def test_budget_tool_dispatch_carries_context_kpis(self, monkeypatch):
        """context_kpis (from kpi_snapshot) must reach the expert params."""
        from src.agent import tools as agent_tools

        captured = {}

        class _FakeExpert:
            async def run(self, state):
                captured.update(state["params"])
                return {"success": True, "expert": "budget",
                        "kpis": {"expected_roi": 1.5}, "allocation": {}}

        monkeypatch.setattr("src.graph.nodes._create_expert", lambda key: _FakeExpert())
        digest = await agent_tools.execute_tool_call_async(
            {"query": "全局预算分配", "budget": 100_000, "kpi_snapshot": {"subsidy_expected_roi": 2.0}},
            "run_budget_analysis",
            {"query": "全局预算分配", "budget": 100_000},
        )
        assert captured["context_kpis"] == {"subsidy_expected_roi": 2.0}
        assert digest["success"] is True
```

（`_run_expert_tool` 内 `_create_expert`/`_validate_expert_result` 是函数内 import，monkeypatch 目标为 `src.graph.nodes`。若 validate 拒绝 FakeExpert 返回结构，则给 FakeExpert 返回补齐 `success/kpis` 字段并接受 `_validated` 缺省路径——实现时以实际行为校准断言。）

**Step 2: 确认失败** — spec is None

**Step 3: 实现** — 上述 3 处修改。描述文案：

```python
"budget": (
    "【用途】全局预算统筹：在投放(ad)/补贴(subsidy)/召回(retention)/券(conversion)四个渠道间"
    "分配总预算。响应曲线边际递减(log1p 饱和)，整数规划求全局最优切分；"
    "若本运行已有其他专家产出 KPI(如补贴 ROI)会自动校准曲线。"
    "【何时调用】问题涉及跨渠道预算分配、统筹、预算怎么分、marketing mix 时。"
    "【参数】query + budget（必须提供，单位 CNY）。"
    "【失败模式】budget 缺失或非正数时报错；无先验专家结果时用默认曲线"
    "（data_source=default_curves）；求解器不可用时精确枚举兜底（method=enumeration）。"
),
```

**Step 4: 确认通过**

**Step 5: Commit** — `git commit -m "feat(budget): register run_budget_analysis chief tool; inject context_kpis from kpi_snapshot"`

---

## Task 7: 评测样本 + 全量回归

**Files:**
- Modify: `src/evals/dataset.py` — `_BUILTIN_SAMPLES` 追加 2 条 budget 样本
- Test: 全量 suite

**Step 1: 追加样本**

```python
{
    "id": "budget_001",
    "input_query": "下季度 50 万预算，在投放、补贴、召回之间怎么分配？",
    "scope": "budget",
    "expected_keys": ["allocation", "channel_details", "expected_roi", "analysis"],
    "reference_answer": "应包含四渠道预算切分、各渠道预期增量GMV与ROI、总期望ROI、以及边际收益递减下的分配理由。",
    "agent_name": "budget",
    "metadata": {"difficulty": "medium", "tags": ["预算分配", "整数规划", "marketing mix"]},
},
{
    "id": "budget_002",
    "input_query": "全局预算统筹：补贴和投放哪个边际收益更高",
    "scope": "budget",
    "expected_keys": ["allocation", "expected_roi", "solver_method"],
    "reference_answer": "应包含各渠道边际收益对比、预算切分方案、求解方法与数据口径标注。",
    "agent_name": "budget",
    "metadata": {"difficulty": "easy", "tags": ["边际收益", "跨渠道"]},
},
```

**Step 2: 全量测试 + 路由评测**

```bash
GPA_DEMO_MODE=true GPA_LLM_API_KEY=sk-test-placeholder GPA_DB_URL=sqlite+aiosqlite:///./test.db \
uv run python -m pytest tests/ -q --tb=short
uv run gpa eval --router
uv run gpa eval --heuristic
```
Expected: 全部 PASS；router eval 不回归（新增样本命中 budget）。

**Step 3: Commit** — `git commit -m "test(budget): eval samples + full-suite regression green"`

---

## Task 8: 文档同步（README / CLAUDE.md / DESIGN）

**Files:**
- Modify: `README.md` — 五领域专家表加 budget 行；架构图专家层 `prospect / conversion / subsidy / retention / ad` → 加 `budget(统筹)`
- Modify: `CLAUDE.md` — "5 experts" → "6 experts"，架构段加一句 budget 定位
- Modify: `docs/DESIGN-v9.md` — 末尾追加 v9.1 小节：全局预算统筹层的设计动机（竖井→统筹）、凹曲线+IP 公式、校准规则、与 KPI 闸门/L1 台账的协同

**Step 1: 修改三处文档**
**Step 2: 复跑 `gpa eval --heuristic` 确认文档变更无副作用**
**Step 3: Commit** — `git commit -m "docs(budget): README/CLAUDE/DESIGN v9.1 — global budget orchestration layer"`

---

## 完成标准（Definition of Done）

1. `pytest tests/` 全绿（511+ 新增 ≈ 26 个用例）
2. `gpa eval --router` / `--heuristic` 不回归
3. 零 key 模式：budget 专家两次运行 kpis/allocation 逐字节一致
4. 纯"补贴预算"查询不误路由到 budget（路由不抢词）
5. 面试一句话：**"五个竖井专家之上补了全局资源观——预算在渠道间按边际收益流动，而不是每个渠道各自最优"**
