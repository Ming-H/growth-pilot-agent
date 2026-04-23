"""Tests for FastAPI authentication dependencies."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.auth.dependencies import get_current_user, require_role
from src.auth.jwt import create_access_token


class TestGetCurrentUser:
    """Tests for the get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self, test_user, db_session):
        """A valid JWT token returns the matching User object."""
        token = create_access_token({
            "sub": test_user.id,
            "org_id": test_user.org_id,
        })

        creds = MagicMock()
        creds.credentials = token

        user = await get_current_user(creds, db_session)
        assert user.id == test_user.id
        assert user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self, db_session):
        """An invalid token raises HTTP 401."""
        creds = MagicMock()
        creds.credentials = "invalid-token"

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds, db_session)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_sub_raises_401(self, db_session):
        """A token without a 'sub' claim raises HTTP 401."""
        token = create_access_token({"org_id": "org-123"})

        creds = MagicMock()
        creds.credentials = token

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds, db_session)
        assert exc_info.value.status_code == 401
        assert "Invalid token payload" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_nonexistent_user_raises_401(self, db_session):
        """A token with a non-existent user ID raises HTTP 401."""
        token = create_access_token({"sub": "nonexistent-user-id"})

        creds = MagicMock()
        creds.credentials = token

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds, db_session)
        assert exc_info.value.status_code == 401
        assert "not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_inactive_user_raises_401(self, test_user, db_session):
        """An inactive user raises HTTP 401 even with a valid token."""
        test_user.is_active = False
        await db_session.commit()

        token = create_access_token({"sub": test_user.id})

        creds = MagicMock()
        creds.credentials = token

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds, db_session)
        assert exc_info.value.status_code == 401
        assert "inactive" in exc_info.value.detail


class TestRequireRole:
    """Tests for the require_role dependency factory."""

    def test_require_role_returns_coroutine_function(self):
        """require_role returns an async function."""
        checker = require_role("admin", "owner")
        import asyncio
        assert asyncio.iscoroutinefunction(checker)

    @pytest.mark.asyncio
    async def test_user_with_matching_role_passes(self, test_user):
        """A user with a matching role passes the check."""
        test_user.role = "admin"
        checker = require_role("admin", "owner")
        result = await checker(test_user)
        assert result.id == test_user.id

    @pytest.mark.asyncio
    async def test_user_with_wrong_role_raises_403(self, test_user):
        """A user without the required role raises HTTP 403."""
        test_user.role = "member"
        checker = require_role("admin", "owner")

        with pytest.raises(HTTPException) as exc_info:
            await checker(test_user)
        assert exc_info.value.status_code == 403
        assert "admin" in exc_info.value.detail
