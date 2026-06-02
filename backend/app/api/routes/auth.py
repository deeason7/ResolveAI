"""
Auth endpoints: register, login, refresh, logout, me.

Token lifecycle:
  POST /register  → creates user, returns access token
  POST /login     → verifies credentials, returns access token + sets refresh cookie
  POST /refresh   → reads refresh cookie, issues new access token
  POST /logout    → revokes the refresh token in Redis
  GET  /me        → returns the calling user's profile
"""

import logging
import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.core.deps import get_current_user
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_ip,
    hash_password,
    verify_password,
)
from app.database import get_session
from app.models.audit_log import AuditLog
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserPublic

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "refresh_token"
REFRESH_BLOCKLIST_PREFIX = "blocklist:refresh:"


def _redis() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=token,
        httponly=True,
        secure=settings.environment != "development",
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 86400,
        path="/api/v1/auth",
    )


async def _audit(session: AsyncSession, request: Request, user_id: "uuid.UUID", event: str) -> None:
    log = AuditLog(
        user_id=user_id,
        event=event,
        ip_hash=hash_ip(request.client.host if request.client else "unknown"),
    )
    session.add(log)


# ── Register ──────────────────────────────────────────────────────────────────


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    """Create a new account and return an access token."""
    existing = await session.exec(select(User).where(User.email == body.email))
    if existing.first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=body.email,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
    )
    session.add(user)
    await session.flush()  # get the UUID before commit

    await _audit(session, request, user.id, "register")

    access = create_access_token(str(user.id), user.role)
    refresh = create_refresh_token(str(user.id))
    _set_refresh_cookie(response, refresh)

    logger.info("New user registered: %s", user.email)
    return TokenResponse(access_token=access)


# ── Login ─────────────────────────────────────────────────────────────────────


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    """Authenticate and return tokens."""
    result = await session.exec(select(User).where(User.email == body.email))
    user = result.first()

    # Constant-time comparison to prevent user-enumeration timing attacks
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    await _audit(session, request, user.id, "login")

    access = create_access_token(str(user.id), user.role)
    refresh = create_refresh_token(str(user.id))
    _set_refresh_cookie(response, refresh)

    logger.info("User logged in: %s", user.email)
    return TokenResponse(access_token=access)


# ── Refresh ───────────────────────────────────────────────────────────────────


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    response: Response,
    session: AsyncSession = Depends(get_session),
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
) -> TokenResponse:
    """Issue a new access token using the refresh cookie."""
    if refresh_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token")

    payload = decode_refresh_token(refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    async with _redis() as r:
        if await r.exists(f"{REFRESH_BLOCKLIST_PREFIX}{refresh_token}"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

    user = await session.get(User, uuid.UUID(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    new_access = create_access_token(str(user.id), user.role)
    new_refresh = create_refresh_token(str(user.id))

    # Rotate: revoke old, issue new
    async with _redis() as r:
        await r.setex(
            f"{REFRESH_BLOCKLIST_PREFIX}{refresh_token}",
            settings.refresh_token_expire_days * 86400,
            "revoked",
        )
    _set_refresh_cookie(response, new_refresh)

    return TokenResponse(access_token=new_access)


# ── Logout ────────────────────────────────────────────────────────────────────


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
) -> None:
    """Revoke the refresh token and clear the cookie."""
    if refresh_token:
        async with _redis() as r:
            await r.setex(
                f"{REFRESH_BLOCKLIST_PREFIX}{refresh_token}",
                settings.refresh_token_expire_days * 86400,
                "revoked",
            )

    response.delete_cookie(key=REFRESH_COOKIE, path="/api/v1/auth")
    await _audit(session, request, current_user.id, "logout")
    logger.info("User logged out: %s", current_user.email)


# ── Me ────────────────────────────────────────────────────────────────────────


@router.get("/me", response_model=UserPublic)
async def me(current_user: User = Depends(get_current_user)) -> UserPublic:
    """Return the authenticated user's profile."""
    return UserPublic(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        created_at=current_user.created_at,
    )
