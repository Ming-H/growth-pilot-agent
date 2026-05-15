"""Pydantic schemas for auth endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str = ""
    org_name: str = ""


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
