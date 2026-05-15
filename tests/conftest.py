"""Shared pytest fixtures for growth-pilot-agent tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.core.config import Settings, get_settings
from src.graph.state import AnalysisState


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings() -> Settings:
    """Create a Settings instance with safe test defaults (no real API keys)."""
    return Settings(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_api_key="sk-test-key-for-testing",
        llm_base_url="https://api.example.com",
        llm_temperature=0.0,
        memory_base_path="./data/test_memory",
        log_level="DEBUG",
    )


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Clear the lru_cache on get_settings before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# State fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Provide a minimal valid state dict for testing."""
    return {
        "query": "帮我分析最近的货运增长数据",
        "data_path": "/data/test/user_logs.csv",
        "errors": [],
        "metadata": [],
    }


@pytest.fixture
def enriched_state(sample_state: dict[str, Any]) -> dict[str, Any]:
    """Provide a state dict matching AnalysisState schema."""
    state = dict(sample_state)
    state["budget"] = 50000.0
    state["scope"] = "full"
    state["expert_results"] = [
        {"expert": "prospect", "user_count": 1200, "success": True},
        {"expert": "subsidy", "expected_roi": 2.5, "success": True},
        {"expert": "ad", "expected_cpa": 45.0, "success": True},
    ]
    state["execution_errors"] = []
    return state


# ---------------------------------------------------------------------------
# Memory / filesystem fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_memory_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for MemoryManager persistence tests."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    return mem_dir


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create a minimal CSV file
    csv_file = data_dir / "test_data.csv"
    csv_file.write_text("user_id,event,date\n1,click,2024-01-01\n2,order,2024-01-02\n")

    # Create a minimal JSON file
    json_file = data_dir / "test_data.json"
    json_file.write_text('{"key": "value", "count": 42}')

    return data_dir


# ---------------------------------------------------------------------------
# Mock LLM fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """Provide a mock LLM that returns a canned response without API calls."""
    llm = MagicMock()
    response = MagicMock()
    response.content = '{"result": "mocked"}'
    llm.ainvoke = MagicMock(return_value=response)
    llm.invoke = MagicMock(return_value=response)
    return llm


# ---------------------------------------------------------------------------
# Environment fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env():
    """Remove GPA_* environment variables for isolated tests."""
    gpa_keys = [k for k in os.environ if k.startswith("GPA_")]
    saved = {k: os.environ.pop(k) for k in gpa_keys}
    yield
    os.environ.update(saved)


# ---------------------------------------------------------------------------
# Database fixtures (in-memory SQLite for async tests)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_engine():
    """Create a test database engine with in-memory SQLite."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from src.db.database import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """Provide a test database session."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def test_org(db_session):
    """Create a test organization."""
    from src.db.models import Organization

    org = Organization(name="Test Org", plan="free")
    db_session.add(org)
    await db_session.commit()
    await db_session.refresh(org)
    return org


@pytest_asyncio.fixture
async def test_user(db_session, test_org):
    """Create a test user."""
    from src.auth.security import hash_password
    from src.db.models import User

    user = User(
        email="test@example.com",
        hashed_password=hash_password("testpass123"),
        name="Test User",
        org_id=test_org.id,
        role="admin",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def auth_token(test_user):
    """Generate a valid JWT token for the test user."""
    from src.auth.jwt import create_access_token

    return create_access_token({
        "sub": test_user.id,
        "org_id": test_user.org_id,
        "role": test_user.role,
    })
