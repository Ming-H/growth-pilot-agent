"""Pydantic schemas for auth endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str = ""
    org_name: str = ""

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        """C9: basic shape check + lowercase normalisation (no new dep —
        ``A@B.com`` and ``a@b.com`` used to be distinct accounts)."""
        import re

        v = v.strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("invalid email address")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    org_id: str
    role: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    org_id: str
    role: str
    is_active: bool

    model_config = {"from_attributes": True}
