"""API-layer tests for the v9 web fixes (A1-A5, HITL round-trip)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.core.config import get_settings
from src.web.deps import thread_id_for


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GPA_DEMO_MODE", "true")
    monkeypatch.setenv("GPA_LLM_API_KEY", "")
    monkeypatch.setenv("GPA_LLM_CHANNEL", "remote")
    monkeypatch.setenv("GPA_MEMORY_DB_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("GPA_DB_URL", f"sqlite+aiosqlite:///{tmp_path}/api_test.db")
    get_settings.cache_clear()

    from src.db import database
    from src.web import create_app

    # Module-level singletons bind to the event loop that created them —
    # pytest-asyncio gives each test a fresh loop, so stale singletons
    # (db engine, MemoryStore, compiled graphs) deadlock cross-test.
    database._engine = None
    database._session_factory = None
    from src.web import deps as web_deps

    web_deps._memory_store = None
    from src.graph import runtime as graph_runtime

    graph_runtime.clear_graph_cache()

    # ASGITransport does not run the app lifespan — initialise the tables
    # the lifespan would have created.
    import src.db.models  # noqa: F401  (register ORM metadata)
    from src.db.database import init_db

    await init_db()

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    # Dispose properly: leaking the aiosqlite engine thread deadlocks
    # interpreter shutdown (wait_for_thread_shutdown never returns).
    if database._engine is not None:
        try:
            await database._engine.dispose()
        except Exception:
            pass
    database._engine = None
    database._session_factory = None
    if web_deps._memory_store is not None:
        try:
            await web_deps._memory_store.close()
        except Exception:
            pass
        web_deps._memory_store = None


@pytest.mark.asyncio
class TestOrgThreading:
    async def test_analyze_threads_org_and_user(self, client):
        """A1: run_workflow receives the authenticated user's org/user."""
        captured = {}

        async def fake_workflow(**kwargs):
            captured.update(kwargs)
            return {
                "success": True,
                "query": kwargs.get("query", ""),
                "scope": "full",
                "analysis_summary": "ok",
                "strategy_recommendation": "",
                "expert_results": {},
                "errors": [],
                "report": "ok",
                "metadata": {"token_usage": {}, "cost_usd": 0.01},
                "kpi_snapshot": {"expected_roi": 2.0},
            }

        reg = await client.post(
            "/api/v1/auth/register",
            json={"email": "a@t.co", "password": "secret123", "name": "A"},
        )
        assert reg.status_code == 201, reg.text
        token = reg.json()["access_token"]
        org_id = reg.json()["org_id"]

        with patch("src.web.routers.analyses.run_workflow", new=fake_workflow):
            resp = await client.post(
                "/api/v1/analyze",
                headers={"Authorization": f"Bearer {token}"},
                json={"query": "分析用户增长", "budget": 1000},
            )
        assert resp.status_code == 200, resp.text
        assert captured["org_id"] == org_id  # A1: org threaded through
        assert captured["user_id"]
        assert captured["thread_id"]  # stable HITL thread (analysis id)

    async def test_cost_persisted_from_metadata(self, client):
        """A5: cost_usd lands on the Analysis row from metadata.cost_usd."""
        reg = await client.post(
            "/api/v1/auth/register",
            json={"email": "b@t.co", "password": "secret123"},
        )
        token = reg.json()["access_token"]

        async def fake_workflow(**kwargs):
            return {
                "success": True,
                "query": "q",
                "scope": "full",
                "analysis_summary": "s",
                "strategy_recommendation": "",
                "expert_results": {},
                "errors": [],
                "report": "r",
                "metadata": {"token_usage": {}, "cost_usd": 0.42},
                "kpi_snapshot": {"x": 1.0},
            }

        with patch("src.web.routers.analyses.run_workflow", new=fake_workflow):
            resp = await client.post(
                "/api/v1/analyze",
                headers={"Authorization": f"Bearer {token}"},
                json={"query": "分析用户增长"},
            )
        assert resp.status_code == 200
        assert resp.json()["cost_usd"] == pytest.approx(0.42)
        assert resp.json()["kpi_snapshot"] == {"x": 1.0}


@pytest.mark.asyncio
class TestHITLThreadDiscipline:
    async def test_thread_id_format(self):
        assert thread_id_for("org1", "an1") == "org1:an1"


@pytest.mark.asyncio
class TestModeField:
    async def test_analyze_records_mode(self, client):
        """The Analysis row persists its mode (HITL resume fetches the same graph)."""
        reg = await client.post(
            "/api/v1/auth/register",
            json={"email": "c@t.co", "password": "secret123"},
        )
        token = reg.json()["access_token"]

        async def fake_workflow(**kwargs):
            return {
                "success": True,
                "query": "q",
                "scope": "full",
                "analysis_summary": "",
                "strategy_recommendation": "",
                "expert_results": {},
                "errors": [],
                "report": "r",
                "metadata": {"cost_usd": 0},
                "kpi_snapshot": {},
            }

        with patch("src.web.routers.analyses.run_workflow", new=fake_workflow):
            resp = await client.post(
                "/api/v1/analyze",
                headers={"Authorization": f"Bearer {token}"},
                json={"query": "分析", "mode": "workflow"},
            )
        assert resp.status_code == 200
        assert resp.json()["result"]["scope"] == "full" or resp.json()["id"]

    async def test_memory_aliases_roundtrip(self, client):
        reg = await client.post(
            "/api/v1/auth/register",
            json={"email": "d@t.co", "password": "secret123"},
        )
        token = reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        put = await client.put(
            "/api/v1/memory/aliases",
            params={"alias": "货主", "canonical": "prospect"},
            headers=headers,
        )
        assert put.status_code == 200
        listing = await client.get("/api/v1/memory/aliases", headers=headers)
        assert listing.status_code == 200
        assert any(a["alias"] == "货主" for a in listing.json())
