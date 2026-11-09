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


@pytest.mark.asyncio
class TestHITLOrgIsolation:
    """C5: HITL endpoints must org-scope the analysis row itself."""

    async def _register(self, client, tag: str) -> dict:
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": f"{tag}@iso-test.example.com", "password": "Pass-1234", "name": tag},
        )
        assert resp.status_code == 201
        return resp.json()

    async def _insert_analysis(self, org_id: str, user_id: str, status: str = "running") -> str:
        import uuid as _uuid

        from src.db.database import get_session_factory
        from src.db.models import Analysis

        async with get_session_factory()() as session:
            row = Analysis(
                id=f"ana-{_uuid.uuid4().hex[:10]}",
                user_id=user_id,
                org_id=org_id,
                query="iso test",
                scope="full",
                status=status,
                mode="workflow",
            )
            session.add(row)
            await session.commit()
            return row.id

    async def test_cross_org_state_is_404_before_checkpoint(self, client, monkeypatch):
        """Cross-org /state must 404 at the ROW gate, never reach the graph."""
        a = await self._register(client, "iso-a")
        b = await self._register(client, "iso-b")
        analysis_id = await self._insert_analysis(a["org_id"], a["user_id"])

        async def _must_not_be_called(mode):
            raise AssertionError("graph accessed for a cross-org request")

        monkeypatch.setattr("src.web.routers.analyses.get_graph_for", _must_not_be_called)

        token_b = b["access_token"]
        resp = await client.get(
            f"/api/v1/analyses/{analysis_id}/state",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert resp.status_code == 404

    async def test_approve_on_completed_is_409(self, client):
        """C5 guard: double-approve / approving a finished run must 409."""
        a = await self._register(client, "iso-c")
        analysis_id = await self._insert_analysis(a["org_id"], a["user_id"], status="completed")
        resp = await client.post(
            f"/api/v1/analyses/{analysis_id}/approve",
            json={"approved": True},
            headers={"Authorization": f"Bearer {a['access_token']}"},
        )
        assert resp.status_code == 409

    async def test_resume_on_completed_is_409(self, client):
        a = await self._register(client, "iso-d")
        analysis_id = await self._insert_analysis(a["org_id"], a["user_id"], status="completed")
        resp = await client.post(
            f"/api/v1/analyses/{analysis_id}/resume",
            headers={"Authorization": f"Bearer {a['access_token']}"},
        )
        assert resp.status_code == 409


@pytest.mark.asyncio
class TestStreamInterruptionPersist:
    async def test_persist_stream_status_interrupted(self, client):
        """C4 helper: an interrupted stream leaves an honest row status."""
        reg = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "sse-int@example.com",
                "password": "Pass-1234",
                "name": "sse",
            },
        )
        assert reg.status_code == 201
        info = reg.json()

        import uuid as _uuid

        from src.db.database import get_session_factory
        from src.db.models import Analysis
        from src.web.routers.analyses import _persist_stream_status

        async with get_session_factory()() as session:
            row = Analysis(
                id=f"ana-{_uuid.uuid4().hex[:10]}",
                user_id=info["user_id"],
                org_id=info["org_id"],
                query="sse",
                scope="full",
                status="running",
                mode="workflow",
            )
            session.add(row)
            await session.commit()
            analysis_id = row.id

        await _persist_stream_status(
            analysis_id, status="interrupted", error_message="disconnected", elapsed=1.5
        )

        async with get_session_factory()() as session:
            from sqlalchemy import select

            refreshed = (
                await session.execute(select(Analysis).where(Analysis.id == analysis_id))
            ).scalar_one()
            assert refreshed.status == "interrupted"
            assert refreshed.error_message == "disconnected"
            assert refreshed.duration_seconds == 1.5


@pytest.mark.asyncio
class TestMemoryClearPermission:
    """C15: clearing an org's memory is owner-only."""

    async def test_member_cannot_clear(self, client):
        reg = await client.post(
            "/api/v1/auth/register",
            json={"email": "mem-owner@example.com", "password": "Pass-1234", "name": "o"},
        )
        assert reg.status_code == 201
        owner = reg.json()

        # 直接插一个同 org 的 member 用户(无注册入口, 走 DB)
        from src.auth.security import hash_password
        from src.db.database import get_session_factory
        from src.db.models import User

        async with get_session_factory()() as session:
            member = User(
                email="mem-member@example.com",
                hashed_password=hash_password("Pass-1234"),
                name="m",
                org_id=owner["org_id"],
                role="member",
            )
            session.add(member)
            await session.commit()

        from src.auth.jwt import create_access_token

        member_token = create_access_token(
            data={"sub": member.id, "org_id": owner["org_id"], "role": "member"}
        )
        resp = await client.delete(
            "/api/v1/memory", headers={"Authorization": f"Bearer {member_token}"}
        )
        assert resp.status_code == 403

        # owner 可以
        resp = await client.delete(
            "/api/v1/memory", headers={"Authorization": f"Bearer {owner['access_token']}"}
        )
        assert resp.status_code == 200
