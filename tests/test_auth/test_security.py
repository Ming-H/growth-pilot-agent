"""Tests for password hashing and JWT token handling."""
from __future__ import annotations

from datetime import timedelta

import pytest

from src.auth.jwt import create_access_token, decode_access_token
from src.auth.security import hash_password, verify_password


class TestPasswordHashing:
    """Tests for password hash and verify functions."""

    def test_hash_produces_different_output(self):
        """Hashing a password produces a different string."""
        hashed = hash_password("mypassword")
        assert hashed != "mypassword"

    def test_hash_unique_per_call(self):
        """Each hash call produces a unique bcrypt hash (due to salt)."""
        h1 = hash_password("samepassword")
        h2 = hash_password("samepassword")
        assert h1 != h2

    def test_verify_correct_password(self):
        """verify_password returns True for the correct password."""
        hashed = hash_password("mypassword")
        assert verify_password("mypassword", hashed) is True

    def test_verify_wrong_password(self):
        """verify_password returns False for an incorrect password."""
        hashed = hash_password("mypassword")
        assert verify_password("wrongpassword", hashed) is False

    def test_verify_empty_password(self):
        """verify_password returns False for an empty password guess."""
        hashed = hash_password("nonempty")
        assert verify_password("", hashed) is False


class TestJWTTokens:
    """Tests for JWT creation and decoding."""

    def test_jwt_roundtrip(self):
        """A created token can be decoded to retrieve claims."""
        token = create_access_token({"sub": "user-123", "org_id": "org-456"})
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "user-123"
        assert payload["org_id"] == "org-456"

    def test_jwt_contains_exp(self):
        """Token payload includes an expiration claim."""
        token = create_access_token({"sub": "user-123"})
        payload = decode_access_token(token)
        assert payload is not None
        assert "exp" in payload

    def test_jwt_expired(self):
        """A token with a past expiration decodes to None."""
        token = create_access_token(
            {"sub": "user-123"}, expires_delta=timedelta(seconds=-1)
        )
        payload = decode_access_token(token)
        assert payload is None

    def test_jwt_invalid_token(self):
        """Decoding a garbage token returns None."""
        payload = decode_access_token("invalid.token.here")
        assert payload is None

    def test_jwt_tampered_token(self):
        """A token with a modified payload returns None."""
        token = create_access_token({"sub": "user-123"})
        # Tamper with the token by changing a character in the middle
        tampered = token[:-5] + "x" + token[-4:]
        payload = decode_access_token(tampered)
        assert payload is None

    def test_jwt_custom_claims(self):
        """Custom claims like role are preserved in the token."""
        token = create_access_token({
            "sub": "user-abc",
            "org_id": "org-def",
            "role": "owner",
        })
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["role"] == "owner"

    def test_jwt_custom_expiry(self):
        """A custom expiry delta is respected."""
        token = create_access_token(
            {"sub": "user-123"}, expires_delta=timedelta(hours=1)
        )
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "user-123"
