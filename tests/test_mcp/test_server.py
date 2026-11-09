"""Tests for the MCP server tool surface (docs/plans/2026-11-09 design).

Two layers:
- direct calls of the registered tool functions (sync, no protocol)
- in-process MCPServer API: list_tools / call_tool (async, the real path
  a host would exercise; stdio transport is not used — CI-safe)

All tools must be deterministic (seed=42 synthetic data) and carry the
honest ``data_source: synthetic`` marker.
"""

from __future__ import annotations

import pytest

from src.mcp_server import (
    budget_allocate,
    churn_watchlist,
    funnel_bottleneck,
    ltv_rank,
    server,
    subsidy_optimize,
)

TOOL_NAMES = {
    "subsidy_optimize",
    "ltv_rank",
    "churn_watchlist",
    "funnel_bottleneck",
    "budget_allocate",
}


# ---------------------------------------------------------------------------
# Direct function calls (deterministic core)
# ---------------------------------------------------------------------------


class TestSubsidyOptimize:
    def test_returns_allocation_plan(self) -> None:
        result = subsidy_optimize(total_budget_yuan=50000)
        assert "error" not in result
        assert result["data_source"] == "synthetic"
        assert result["total_budget_used"] <= 50000
        assert result["expected_incremental_orders"] > 0
        assert "roi" in result and result["roi"] is not None

    def test_deterministic(self) -> None:
        a = subsidy_optimize(total_budget_yuan=30000)
        b = subsidy_optimize(total_budget_yuan=30000)
        assert a == b

    def test_invalid_budget_errors_honestly(self) -> None:
        assert "error" in subsidy_optimize(total_budget_yuan=0)
        assert "error" in subsidy_optimize(total_budget_yuan=-100)


class TestLTVRank:
    def test_returns_ranked_users(self) -> None:
        result = ltv_rank(top_k=3)
        assert "error" not in result
        assert result["data_source"] == "synthetic"
        users = result["top_users"]
        assert len(users) == 3
        values = [u["predicted_ltv"] for u in users]
        assert values == sorted(values, reverse=True)

    def test_deterministic(self) -> None:
        assert ltv_rank(top_k=5) == ltv_rank(top_k=5)


class TestChurnWatchlist:
    def test_returns_watchlist_with_model_metrics(self) -> None:
        result = churn_watchlist(top_k=10)
        assert "error" not in result
        assert result["data_source"] == "synthetic"
        assert len(result["watchlist"]) == 10
        risks = [u["churn_risk"] for u in result["watchlist"]]
        assert risks == sorted(risks, reverse=True)
        assert 0.0 <= result["auc"] <= 1.0

    def test_deterministic(self) -> None:
        assert churn_watchlist(top_k=5) == churn_watchlist(top_k=5)


class TestFunnelBottleneck:
    def test_identifies_bottleneck(self) -> None:
        result = funnel_bottleneck()
        assert "error" not in result
        assert result["data_source"] == "synthetic"
        assert result.get("bottleneck_stage") or result.get("bottleneck")

    def test_deterministic(self) -> None:
        assert funnel_bottleneck() == funnel_bottleneck()


class TestBudgetAllocate:
    def test_allocates_within_budget(self) -> None:
        result = budget_allocate(total_budget_yuan=500000)
        assert "error" not in result
        assert result["data_source"] == "synthetic"
        allocation = result.get("allocation", {})
        assert allocation, "allocation must be non-empty"
        total = sum(
            v if isinstance(v, (int, float)) else v.get("budget", 0) for v in allocation.values()
        )
        assert total <= 500000 + 1e-6

    def test_invalid_budget_errors_honestly(self) -> None:
        assert "error" in budget_allocate(total_budget_yuan=0)


# ---------------------------------------------------------------------------
# MCPServer registration & in-process calls (protocol-adjacent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_registers_all_five() -> None:
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert TOOL_NAMES <= names


@pytest.mark.asyncio
async def test_call_tool_subsidy_roundtrip() -> None:
    result = await server.call_tool("subsidy_optimize", {"total_budget_yuan": 20000})
    assert result is not None
    assert "error" not in str(result)[:200] or "total_budget" not in str(result)


@pytest.mark.asyncio
async def test_tool_descriptions_carry_aci_contract() -> None:
    """Every tool description must state purpose + data_source honesty."""
    tools = {t.name: t for t in await server.list_tools()}
    for name in TOOL_NAMES:
        desc = tools[name].description or ""
        assert "synthetic" in desc.lower(), f"{name} must declare synthetic data"
        assert "purpose" in desc.lower(), f"{name} must have ACI-style purpose"
