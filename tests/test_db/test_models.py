"""Tests for database ORM models."""
from __future__ import annotations

import pytest

from src.auth.security import hash_password
from src.db.models import Analysis, MemoryEntry, Organization, User


class TestOrganizationModel:
    """Tests for the Organization ORM model."""

    @pytest.mark.asyncio
    async def test_create_organization(self, db_session):
        """Organization can be created with required fields."""
        org = Organization(name="Test Company", plan="pro", monthly_quota=1000)
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        assert org.id is not None
        assert len(org.id) == 36  # UUID format
        assert org.name == "Test Company"
        assert org.plan == "pro"
        assert org.monthly_quota == 1000

    @pytest.mark.asyncio
    async def test_organization_defaults(self, db_session):
        """Organization fields have sensible defaults."""
        org = Organization(name="Defaults Corp")
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        assert org.plan == "free"
        assert org.monthly_quota == 100
        assert org.usage_count == 0
        assert org.settings == {}
        assert org.created_at is not None

    @pytest.mark.asyncio
    async def test_organization_settings_json(self, db_session):
        """Organization settings field stores JSON data."""
        org = Organization(
            name="Settings Corp",
            settings={"theme": "dark", "notifications": True},
        )
        db_session.add(org)
        await db_session.commit()
        await db_session.refresh(org)

        assert org.settings["theme"] == "dark"
        assert org.settings["notifications"] is True


class TestUserModel:
    """Tests for the User ORM model."""

    @pytest.mark.asyncio
    async def test_create_user(self, test_org, db_session):
        """User can be created with required fields."""
        user = User(
            email="new@example.com",
            hashed_password=hash_password("pass123"),
            name="New User",
            org_id=test_org.id,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        assert user.id is not None
        assert user.email == "new@example.com"
        assert user.org_id == test_org.id

    @pytest.mark.asyncio
    async def test_user_defaults(self, test_org, db_session):
        """User fields have sensible defaults."""
        user = User(
            email="defaults@example.com",
            hashed_password=hash_password("pass"),
            org_id=test_org.id,
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        assert user.is_active is True
        assert user.role == "member"
        assert user.name == ""
        assert user.created_at is not None
        assert user.updated_at is not None

    @pytest.mark.asyncio
    async def test_user_org_relationship(self, test_user, test_org, db_session):
        """User.org relationship resolves to the parent Organization."""
        await db_session.refresh(test_user, ["org"])
        assert test_user.org.id == test_org.id
        assert test_user.org.name == test_org.name

    @pytest.mark.asyncio
    async def test_user_role_admin(self, test_user, db_session):
        """User role can be set to admin."""
        assert test_user.role == "admin"

    @pytest.mark.asyncio
    async def test_user_password_is_hashed(self, test_user, db_session):
        """Stored password is a bcrypt hash, not plaintext."""
        assert test_user.hashed_password != "testpass123"
        assert test_user.hashed_password.startswith("$2b$")


class TestAnalysisModel:
    """Tests for the Analysis ORM model."""

    @pytest.mark.asyncio
    async def test_create_analysis(self, test_user, db_session):
        """Analysis can be created with all fields."""
        analysis = Analysis(
            user_id=test_user.id,
            org_id=test_user.org_id,
            query="Analyze weekly growth data",
            scope="full",
            status="completed",
            result={"kpi": {"mau": 50000}},
        )
        db_session.add(analysis)
        await db_session.commit()
        await db_session.refresh(analysis)

        assert analysis.id is not None
        assert analysis.query == "Analyze weekly growth data"
        assert analysis.result["kpi"]["mau"] == 50000

    @pytest.mark.asyncio
    async def test_analysis_defaults(self, test_user, db_session):
        """Analysis fields have sensible defaults."""
        analysis = Analysis(
            user_id=test_user.id,
            org_id=test_user.org_id,
            query="Test query",
        )
        db_session.add(analysis)
        await db_session.commit()
        await db_session.refresh(analysis)

        assert analysis.scope == "full"
        assert analysis.budget == 0.0
        assert analysis.status == "pending"
        assert analysis.result is None
        assert analysis.kpi_snapshot is None
        assert analysis.token_usage == {}
        assert analysis.cost_usd == 0.0
        assert analysis.duration_seconds == 0.0
        assert analysis.error_message is None
        assert analysis.completed_at is None

    @pytest.mark.asyncio
    async def test_analysis_user_relationship(self, test_user, db_session):
        """Analysis.user relationship resolves to the owner User."""
        analysis = Analysis(
            user_id=test_user.id,
            org_id=test_user.org_id,
            query="Relationship test",
        )
        db_session.add(analysis)
        await db_session.commit()
        await db_session.refresh(analysis, ["user"])

        assert analysis.user.id == test_user.id

    @pytest.mark.asyncio
    async def test_analysis_status_lifecycle(self, test_user, db_session):
        """Analysis status can be updated through lifecycle states."""
        analysis = Analysis(
            user_id=test_user.id,
            org_id=test_user.org_id,
            query="Lifecycle test",
        )
        db_session.add(analysis)
        await db_session.commit()

        for status in ("running", "completed"):
            analysis.status = status
            await db_session.commit()
            await db_session.refresh(analysis)
            assert analysis.status == status


class TestMemoryEntryModel:
    """Tests for the MemoryEntry ORM model."""

    @pytest.mark.asyncio
    async def test_create_memory_entry(self, test_org, db_session):
        """MemoryEntry can be created with required fields."""
        entry = MemoryEntry(
            org_id=test_org.id,
            query="Test query",
            scope="full",
            summary="A summary of analysis results",
        )
        db_session.add(entry)
        await db_session.commit()
        await db_session.refresh(entry)

        assert entry.id is not None
        assert entry.org_id == test_org.id
        assert entry.query == "Test query"
        assert entry.scope == "full"
        assert entry.summary == "A summary of analysis results"

    @pytest.mark.asyncio
    async def test_memory_entry_defaults(self, test_org, db_session):
        """MemoryEntry JSON fields have default values."""
        entry = MemoryEntry(
            org_id=test_org.id,
            query="Defaults test",
            scope="prospect",
            summary="Summary",
        )
        db_session.add(entry)
        await db_session.commit()
        await db_session.refresh(entry)

        assert entry.results_summary == {}
        assert entry.relevance_tags == []
        assert entry.created_at is not None
