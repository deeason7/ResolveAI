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
from app.core.deps import get_current_user, get_redis
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_ip,
    hash_password,
    hash_token,
    token_subject,
    verify_password_or_dummy,
)
from app.database import get_session
from app.middleware.rate_limit import limiter
from app.models.audit_log import AuditLog
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserPublic

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "refresh_token"
REFRESH_BLOCKLIST_PREFIX = "blocklist:refresh:"

# Credential endpoints are the expensive ones: every attempt costs a bcrypt
# verify (~180ms of CPU), so the global 200/min default is really a licence to
# saturate a small box. Note this keys on the peer address, which behind a
# managed proxy is the proxy — so treat it as a cost ceiling for the endpoint
# rather than true per-client fairness.
_CREDENTIAL_RATE_LIMIT = "20/minute"


def _blocklist_key(token: str) -> str:
    return f"{REFRESH_BLOCKLIST_PREFIX}{hash_token(token)}"


async def _is_revoked(r: aioredis.Redis, token: str) -> bool:
    """True if this refresh token has been revoked.

    Checks the legacy raw-token key as well: entries written before the switch
    to hashed keys are still live for up to the refresh TTL, and a revoked token
    silently becoming valid again is the one outcome worth a second lookup.
    """
    if await r.exists(_blocklist_key(token)):
        return True
    return bool(await r.exists(f"{REFRESH_BLOCKLIST_PREFIX}{token}"))


async def _revoke(r: aioredis.Redis, token: str) -> None:
    await r.setex(
        _blocklist_key(token),
        settings.refresh_token_expire_days * 86400,
        "revoked",
    )


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
@limiter.limit(_CREDENTIAL_RATE_LIMIT)
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

    access = create_access_token(str(user.id), user.role.value)
    refresh = create_refresh_token(str(user.id))
    _set_refresh_cookie(response, refresh)

    logger.info("New user registered: %s", user.email)
    return TokenResponse(access_token=access)


# ── Login ─────────────────────────────────────────────────────────────────────


@router.post("/login", response_model=TokenResponse)
@limiter.limit(_CREDENTIAL_RATE_LIMIT)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    """Authenticate and return tokens."""
    result = await session.exec(select(User).where(User.email == body.email))
    user = result.first()

    # Always pay for a bcrypt verify, even with no user to verify against —
    # `user is None or not verify_password(...)` would short-circuit, and the
    # ~180ms gap between "no such email" and "wrong password" is a readable
    # account-enumeration oracle over the network.
    if not verify_password_or_dummy(body.password, user.hashed_password if user else None):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    await _audit(session, request, user.id, "login")

    access = create_access_token(str(user.id), user.role.value)
    refresh = create_refresh_token(str(user.id))
    _set_refresh_cookie(response, refresh)

    logger.info("User logged in: %s", user.email)
    return TokenResponse(access_token=access)


# ── Refresh ───────────────────────────────────────────────────────────────────


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    response: Response,
    session: AsyncSession = Depends(get_session),
    r: aioredis.Redis = Depends(get_redis),
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

    if await _is_revoked(r, refresh_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

    subject = token_subject(payload)
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    user = await session.get(User, subject)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    new_access = create_access_token(str(user.id), user.role)
    new_refresh = create_refresh_token(str(user.id))

    # Rotate: revoke old, issue new
    await _revoke(r, refresh_token)
    _set_refresh_cookie(response, new_refresh)

    return TokenResponse(access_token=new_access)


# ── Logout ────────────────────────────────────────────────────────────────────


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
    r: aioredis.Redis = Depends(get_redis),
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
) -> None:
    """Revoke the refresh token and clear the cookie."""
    if refresh_token:
        await _revoke(r, refresh_token)

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
