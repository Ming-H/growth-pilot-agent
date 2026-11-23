"""DeepAgent v10.1 业界水准测试——五项能力逐项验证。

覆盖：ACI 工具规格渲染、DomainContext 构建与约束传递、自检（NaN/极值/负值）、
步级观测完整性、幂等缓存、三个领域 DeepAgent 的领域属性。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.deep.base import DeepAgentBase, DomainContext
from src.agents.deep.prospect import ProspectDeepAgent
from src.agents.deep.retention import RetentionDeepAgent
from src.agents.deep.subsidy import SubsidyDeepAgent
from src.core.tool_spec import ALL_TOOL_SPECS, SUBSIDY_TOOLS


def _mock_llm(actions: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    resps = []
    for act in actions:
        r = MagicMock()
        r.content = f'{{"action": "{act}", "thought": "t", "params": {{}}}}'
        resps.append(r)
    llm.ainvoke.side_effect = resps
    return llm


# ── ACI 工具规格 ──────────────────────────────────────────────────────────


class TestToolSpec:
    def test_spec_has_all_four_elements(self):
        """每个 ToolSpec 必须有 ACI 四要素。"""
        for spec in ALL_TOOL_SPECS.values():
            assert spec.name, f"{spec} missing name"
            assert spec.purpose, f"{spec.name} missing purpose"
            assert spec.when_to_call, f"{spec.name} missing when_to_call"
            assert isinstance(spec.failure_modes, list), f"{spec.name} missing failure_modes"

    def test_render_for_llm_includes_key_info(self):
        spec = SUBSIDY_TOOLS["subsidy_optimize"]
        rendered = spec.render_for_llm()
        assert "subsidy_optimize" in rendered
        assert "ATE" in rendered
        assert "调用时机" in rendered
        assert "已知失败" in rendered

    def test_subsidy_tools_registered(self):
        assert "subsidy_optimize" in SUBSIDY_TOOLS
        assert "elasticity_estimate" in SUBSIDY_TOOLS


# ── DomainContext ─────────────────────────────────────────────────────────


class TestDomainContext:
    def test_default_values(self):
        ctx = DomainContext()
        assert ctx.task == ""
        assert ctx.budget is None
        assert ctx.sibling_kpis == {}
        assert ctx.prior_findings == []
        assert ctx.domain_constraints == []

    def test_with_constraints(self):
        ctx = DomainContext(task="test", budget=5000.0, domain_constraints=["不能为负"])
        assert ctx.budget == 5000.0
        assert "不能为负" in ctx.domain_constraints


# ── Agent 级自检 ─────────────────────────────────────────────────────────


class SimpleDeepAgent(DeepAgentBase):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._test_kpis: dict = {}

    @property
    def domain(self):
        return "test"

    async def _execute_action(self, action, params):
        return "ok"

    async def _compile_final(self, params):
        return {"kpis": self._test_kpis, "summary": "test"}


class TestSelfCheck:
    @pytest.mark.asyncio
    async def test_pass_with_valid_kpis(self):
        agent = SimpleDeepAgent(llm=None, max_deep_steps=1)
        agent._test_kpis = {"ate": 0.08, "budget": 50000}
        result = await agent.deep_analyze({})
        assert result["self_check"]["status"] == "pass"

    @pytest.mark.asyncio
    async def test_warn_with_nan(self):
        agent = SimpleDeepAgent(llm=None, max_deep_steps=1)
        agent._test_kpis = {"ate": float("nan")}
        result = await agent.deep_analyze({})
        assert result["self_check"]["status"] == "warn"
        assert any("NaN" in i for i in result["self_check"]["issues"])

    @pytest.mark.asyncio
    async def test_warn_with_extreme_value(self):
        agent = SimpleDeepAgent(llm=None, max_deep_steps=1)
        agent._test_kpis = {"revenue": 1e15}
        result = await agent.deep_analyze({})
        assert result["self_check"]["status"] == "warn"

    @pytest.mark.asyncio
    async def test_degradation_marker_on_warn(self):
        agent = SimpleDeepAgent(llm=None, max_deep_steps=1)
        agent._test_kpis = {"x": float("inf")}
        result = await agent.deep_analyze({})
        assert "deep:self-check-warn" in result.get("degradations", [])


# ── 步级观测完整性 ────────────────────────────────────────────────────────


class TestStepObservations:
    @pytest.mark.asyncio
    async def test_step_has_five_fields(self):
        llm = _mock_llm(["run_pipeline", "finalize"])
        agent = SimpleDeepAgent(llm=llm, max_deep_steps=3)
        agent._test_kpis = {"x": 1.0}
        result = await agent.deep_analyze({"task": "t"})
        for step in result["deep_steps"]:
            for field in ("thought", "action", "obs", "gate", "ms"):
                assert field in step, f"step missing {field}"

    @pytest.mark.asyncio
    async def test_idempotent_cache_prevents_duplicate(self):
        llm = _mock_llm(["run_pipeline", "run_pipeline", "finalize"])
        agent = SimpleDeepAgent(llm=llm, max_deep_steps=5)
        agent._test_kpis = {"x": 1.0}
        result = await agent.deep_analyze({"task": "t"})
        # 第二个相同 action+params 应被标为 duplicate
        has_dup = any(s["gate"] == "duplicate" for s in result["deep_steps"])
        assert has_dup or result["deep_step_count"] <= 2  # duplicate 应触发收手


# ── 三个领域 DeepAgent ────────────────────────────────────────────────────


class TestDomainAgents:
    def test_subsidy_domain_and_tools(self):
        agent = SubsidyDeepAgent(llm=None)
        assert agent.domain == "subsidy"
        assert "subsidy_optimize" in agent.tool_specs
        assert len(agent._domain_constraints()) >= 3

    def test_prospect_domain_and_tools(self):
        agent = ProspectDeepAgent(llm=None)
        assert agent.domain == "prospect"
        assert "intent_predict" in agent.tool_specs
        assert any("0, 100" in c for c in agent._domain_constraints())

    def test_retention_domain_and_tools(self):
        agent = RetentionDeepAgent(llm=None)
        assert agent.domain == "retention"
        assert "churn_predict" in agent.tool_specs
        assert any("0, 1" in c for c in agent._domain_constraints())

    @pytest.mark.asyncio
    async def test_prospect_zero_llm(self):
        agent = ProspectDeepAgent(llm=None, max_deep_steps=2)
        result = await agent.deep_analyze({})
        assert result.get("deep_domain") == "prospect"
        assert "self_check" in result

    @pytest.mark.asyncio
    async def test_retention_zero_llm(self):
        agent = RetentionDeepAgent(llm=None, max_deep_steps=2)
        result = await agent.deep_analyze({})
        assert result.get("deep_domain") == "retention"
        assert "self_check" in result
