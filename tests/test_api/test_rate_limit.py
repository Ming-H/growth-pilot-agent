"""C1: rate limiting is actually enforced (was dead code — the Limiter
existed but SlowAPIMiddleware was never mounted and no route carried a
limit; /login could be brute-forced without bound)."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.db.database import Base, get_db
from src.web import create_app
from src.web.rate_limit import HAS_SLOWAPI, limiter

pytestmark = [
    pytest.mark.skipif(not HAS_SLOWAPI, reason="slowapi not installed"),
    pytest.mark.asyncio,
]


@pytest_asyncio.fixture
async def rl_app():
    """Fresh app + in-memory DB; limiter ENABLED for this module only."""
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
    limiter.reset()
    limiter.enabled = True
    yield application
    limiter.enabled = False
    limiter.reset()
    application.dependency_overrides.clear()
    await engine.dispose()


class TestRateLimitEnforced:
    async def test_middleware_mounted(self, rl_app):
        from slowapi.middleware import SlowAPIMiddleware

        classes = [m.cls for m in rl_app.user_middleware]
        assert SlowAPIMiddleware in classes

    async def test_login_brute_force_gets_429(self, rl_app):
        """6th login attempt within a minute → 429 (limit 5/minute)."""
        email = f"rl-{uuid.uuid4().hex[:8]}@example.com"
        async with AsyncClient(
            transport=ASGITransport(app=rl_app), base_url="http://test"
        ) as client:
            reg = await client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": "secret-Pass1", "name": "rl"},
            )
            assert reg.status_code == 201

            statuses = [
                (
                    await client.post(
                        "/api/v1/auth/login",
                        json={"email": email, "password": "wrong-pass"},
                    )
                ).status_code
                for _ in range(6)
            ]

        assert statuses[:5] == [401] * 5  # wrong password: auth failure
        assert statuses[5] == 429  # ...but the 6th is throttled

    async def test_analyze_route_carries_limit(self, rl_app):
        """The abuse-prone endpoints are decorated (registered in the limiter)."""
        from slowapi import Limiter

        assert isinstance(limiter, Limiter)
        keys = set(limiter._route_limits.keys())
        assert any(k.endswith("auth.login") for k in keys), keys
        assert any(k.endswith("auth.register") for k in keys), keys
        assert any(k.endswith("analyses.analyze") for k in keys), keys
        assert any(k.endswith("analyses.analyze_stream") for k in keys), keys
