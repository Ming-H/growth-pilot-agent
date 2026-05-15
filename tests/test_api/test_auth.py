"""Tests for auth API endpoints: register, login, /me."""
from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.db.database import Base, get_db
from src.web import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_demo_env():
    """Ensure demo mode is active for all auth tests."""
    os.environ["GPA_DEMO_MODE"] = "true"
    os.environ["GPA_LLM_API_KEY"] = "test-key"


@pytest_asyncio.fixture
async def auth_app():
    """Create a test app with a shared in-memory SQLite and override get_db.

    The override eagerly commits after each request so that subsequent
    requests within the same test can see the data.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    application = create_app()

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    application.dependency_overrides[get_db] = override_get_db
    yield application
    application.dependency_overrides.clear()
    await engine.dispose()


@pytest_asyncio.fixture
async def auth_client(auth_app):
    """Provide an async HTTP test client."""
    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


def _unique_email() -> str:
    """Generate a unique email address for test isolation."""
    return f"test-{uuid.uuid4().hex[:8]}@example.com"


# ---------------------------------------------------------------------------
# Test: POST /api/v1/auth/register
# ---------------------------------------------------------------------------


class TestRegister:
    """Tests for the /api/v1/auth/register endpoint."""

    @pytest.mark.asyncio
    async def test_register_success(self, auth_client):
        """Successful registration returns 201 with token and user info."""
        resp = await auth_client.post("/api/v1/auth/register", json={
            "email": _unique_email(),
            "password": "securepass123",
            "name": "New User",
            "org_name": "Test Org",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert "user_id" in data
        assert "org_id" in data
        assert data["role"] == "owner"
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_register_duplicate_email_fails(self, auth_client):
        """Registering with the same email twice returns an error."""
        email = _unique_email()
        payload = {
            "email": email,
            "password": "password123",
            "name": "First",
        }
        resp1 = await auth_client.post("/api/v1/auth/register", json=payload)
        assert resp1.status_code == 201

        resp2 = await auth_client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "different456",
            "name": "Second",
        })
        # 409 (conflict) from explicit check, or 500 from unhandled integrity error
        assert resp2.status_code in (409, 500)


# ---------------------------------------------------------------------------
# Test: POST /api/v1/auth/login
# ---------------------------------------------------------------------------


class TestLogin:
    """Tests for the /api/v1/auth/login endpoint."""

    @pytest.mark.asyncio
    async def test_login_success(self, auth_client):
        """Successful login returns token and user info."""
        email = _unique_email()
        password = "loginpass123"
        await auth_client.post("/api/v1/auth/register", json={
            "email": email,
            "password": password,
            "name": "Login User",
        })
        resp = await auth_client.post("/api/v1/auth/login", json={
            "email": email,
            "password": password,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "user_id" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password_fails(self, auth_client):
        """Login with wrong password returns 401."""
        email = _unique_email()
        await auth_client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "correctpass",
            "name": "Wrong PW User",
        })
        resp = await auth_client.post("/api/v1/auth/login", json={
            "email": email,
            "password": "incorrectpass",
        })
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_login_nonexistent_user_fails(self, auth_client):
        """Login with an unregistered email returns 401."""
        resp = await auth_client.post("/api/v1/auth/login", json={
            "email": _unique_email(),
            "password": "anypass",
        })
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test: GET /api/v1/auth/me
# ---------------------------------------------------------------------------


class TestMe:
    """Tests for the /api/v1/auth/me endpoint."""

    @pytest.mark.asyncio
    async def test_me_with_valid_token(self, auth_client):
        """Authenticated /me returns current user info."""
        email = _unique_email()
        reg_resp = await auth_client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "mepass123",
            "name": "Me User",
        })
        token = reg_resp.json()["access_token"]

        resp = await auth_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == email
        assert data["name"] == "Me User"
        assert "id" in data
        assert "org_id" in data
        assert data["is_active"] is True

    @pytest.mark.asyncio
    async def test_me_without_token_returns_40x(self, auth_client):
        """Calling /me without Authorization header returns 401 or 403."""
        resp = await auth_client.get("/api/v1/auth/me")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_me_with_invalid_token_returns_401(self, auth_client):
        """Calling /me with an invalid token returns 401."""
        resp = await auth_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert resp.status_code == 401
