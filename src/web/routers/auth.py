"""Auth endpoints: register / login / me."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.dependencies import get_current_user
from src.auth.jwt import create_access_token
from src.auth.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from src.auth.security import hash_password, verify_password
from src.db.database import get_db
from src.db.models import Organization, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(
    req: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Create a new user and organization, return JWT token."""
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    org_name = req.org_name or f"{req.name or req.email.split('@')[0]}'s Org"
    org = Organization(name=org_name)
    db.add(org)
    await db.flush()

    user = User(
        email=req.email,
        hashed_password=hash_password(req.password),
        name=req.name,
        org_id=org.id,
        role="owner",
    )
    db.add(user)
    await db.flush()

    token = create_access_token(data={"sub": user.id, "org_id": org.id, "role": user.role})

    return TokenResponse(
        access_token=token,
        user_id=user.id,
        org_id=org.id,
        role=user.role,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    req: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate user and return JWT token."""
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    token = create_access_token(data={"sub": user.id, "org_id": user.org_id, "role": user.role})

    return TokenResponse(
        access_token=token,
        user_id=user.id,
        org_id=user.org_id,
        role=user.role,
    )


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)) -> UserResponse:
    """Return current authenticated user info."""
    return UserResponse.model_validate(user)
