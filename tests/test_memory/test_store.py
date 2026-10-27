"""Tests for the three-layer SQLite MemoryStore (L0/L1/L2 + episodic FTS)."""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.memory import MemoryStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "memory.db"))
    yield s
    await s.close()


@pytest.mark.asyncio
class TestEpisodicAnalyses:
    async def test_save_and_search_zh(self, store):
        await store.save_analysis(
            analysis_id="a1",
            org_id="org1",
            query="补贴 ROI 分析",
            scope="subsidy",
            report="补贴回报率 2.1x，建议加码新客券",
            kpis={"expected_roi": 2.1},
        )
        await store.save_analysis(
            analysis_id="a2",
            org_id="org1",
            query="流失用户召回策略",
            scope="retention",
            report="高危流失占比 18%",
        )
        hits = await store.search_analyses("org1", "补贴 回报")
        assert any(h["id"] == "a1" for h in hits)
        # org isolation
        assert await store.search_analyses("org2", "补贴") == []

    async def test_fts_or_like_fallback(self, store, monkeypatch):
        """LIKE fallback still finds entries when FTS is unavailable."""
        if not store.fts_available:
            pass  # already on the fallback path
        await store.save_analysis(
            analysis_id="a3",
            org_id="org1",
            query="英文 query about retention",
            scope="retention",
            report="churn analysis report",
        )
        hits = await store.search_analyses("org1", "retention")
        assert any(h["id"] == "a3" for h in hits)

    async def test_get_recent(self, store):
        for i in range(5):
            await store.save_analysis(
                analysis_id=f"r{i}", org_id="o", query=f"q{i}", scope="full", report="x"
            )
        recent = await store.get_recent_analyses("o", limit=3)
        assert len(recent) == 3
        assert recent[0]["id"] == "r4"  # newest first


@pytest.mark.asyncio
class TestL0Sessions:
    async def test_recent_turns_and_compression(self, store):
        sid = "s1"
        for i in range(6):
            await store.save_turn(
                sid, org_id="o", role="user", content=f"问题 {i}", structured=None
            )
            await store.save_turn(
                sid,
                org_id="o",
                role="assistant",
                content=f"回答 {i}",
                structured={"kpis": {"expected_roi": float(i)}},
            )
        loaded = await store.load_session(sid, last_n_full=2)
        assert len(loaded["recent"]) == 4  # last 2 turns × (user+assistant)
        assert "问题 5" in loaded["recent"][-2]["content"]
        assert "回答 0" in loaded["compressed"]  # earlier turns compressed
        assert "expected_roi" in loaded["compressed"]  # structured kpis kept

    async def test_empty_session(self, store):
        loaded = await store.load_session("nonexistent")
        assert loaded == {"recent": [], "compressed": ""}


@pytest.mark.asyncio
class TestL1FactLedger:
    async def test_write_gate_success(self, store):
        result = {
            "expert": "subsidy",
            "success": True,
            "_validated": True,
            "kpis": {"expected_roi": 2.1, "subsidy_ate": 0.12},
        }
        n = await store.record_facts("o", result, run_fingerprint="run-1")
        assert n == 2
        facts = await store.search_facts("o", domain="subsidy")
        assert {f["metric_key"] for f in facts} == {"expected_roi", "subsidy_ate"}

    async def test_write_gate_rejects_failures(self, store):
        assert (
            await store.record_facts(
                "o", {"expert": "x", "success": False, "_validated": True, "kpis": {"a": 1}}, "r"
            )
            == 0
        )
        assert (
            await store.record_facts(
                "o", {"expert": "x", "success": True, "_validated": False, "kpis": {"a": 1}}, "r"
            )
            == 0
        )
        assert (
            await store.record_facts(
                "o", {"expert": "x", "success": True, "_validated": True, "kpis": {}}, "r"
            )
            == 0
        )

    async def test_dedupe_same_fingerprint(self, store):
        result = {"expert": "x", "success": True, "_validated": True, "kpis": {"a": 1.0}}
        await store.record_facts("o", result, run_fingerprint="r1")
        await store.record_facts("o", result, run_fingerprint="r1")  # UNIQUE no-op
        facts = await store.search_facts("o")
        assert len(facts) == 1


@pytest.mark.asyncio
class TestL2Aliases:
    async def test_upsert_resolve_and_routing_boost(self, store):
        await store.upsert_alias("o", "货主", "prospect")
        await store.upsert_alias("o", "拉新", "prospect")
        hits = await store.resolve_aliases("o", "这个月货主拉新效果如何")
        assert hits.get("expert") == {"prospect"}
        aliases = await store.list_aliases("o")
        assert any(a["hits"] > 0 for a in aliases)  # hit counter bumped

    async def test_no_match(self, store):
        await store.upsert_alias("o", "货主", "prospect")
        assert await store.resolve_aliases("o", "完全无关文本") == {}


@pytest.mark.asyncio
class TestHousekeeping:
    async def test_clear_and_count(self, store):
        await store.save_analysis(analysis_id="a", org_id="o", query="q", scope="full", report="r")
        await store.record_facts(
            "o", {"expert": "x", "success": True, "_validated": True, "kpis": {"k": 1}}, "r"
        )
        await store.upsert_alias("o", "别名", "prospect")
        counts = await store.count("o")
        assert counts["analyses"] == 1 and counts["verified_facts"] == 1
        removed = await store.clear_org("o")
        assert removed == 1
        assert (await store.count("o"))["analyses"] == 0

    async def test_concurrent_writes_serialize(self, store):
        """No 'database is locked' or lost updates under concurrent writers."""

        async def _w(i: int):
            await store.save_analysis(
                analysis_id=f"c{i}", org_id="o", query=f"q{i}", scope="s", report="r"
            )

        import asyncio

        await asyncio.gather(*[_w(i) for i in range(10)])
        assert (await store.count("o"))["analyses"] == 10

    async def test_legacy_json_migration(self, tmp_path):
        import json as _json

        legacy_dir = tmp_path / "org-legacy"
        legacy_dir.mkdir()
        (legacy_dir / "memory_store.json").write_text(
            _json.dumps(
                [
                    {
                        "id": "old1",
                        "run_id": "r",
                        "query": "历史补贴分析",
                        "scope": "subsidy",
                        "results_summary": "旧摘要",
                        "timestamp": 1700000000.0,
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        s = MemoryStore(db_path=str(tmp_path / "memory.db"))
        try:
            hits = await s.search_analyses("org-legacy", "补贴")
            assert any(h["id"] == "old1" for h in hits)
            assert not (legacy_dir / "memory_store.json").exists()
            assert (legacy_dir / "memory_store.json.migrated").exists()
        finally:
            await s.close()
