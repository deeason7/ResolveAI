"""
Auth endpoints: register, login, refresh, logout, me.

Token lifecycle:
  POST /register  → creates user, returns access token
  POST /login     → verifies credentials, returns access token + sets refresh cookie
  POST /refresh   → reads refresh cookie, issues new access token
  POST /logout    → revokes the refresh token in Redis
  GET  /me        → returns the calling user's profile
"""

import hmac
import logging
import math
import time
import uuid

import redis.asyncio as aioredis
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from redis.exceptions import RedisError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.core.deps import ensure_demo_is_viewer, get_current_user, get_redis
from app.core.security import (
    create_access_token,
    create_refresh_token,
    credential_fingerprint,
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
from app.models.user import User, UserRole
from app.schemas.auth import (
    AuthCapabilities,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserPublic,
)
from app.services.auth_rate_limit import check_account_limit

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


async def _consume(r: aioredis.Redis, token: str, payload: dict) -> bool:
    """One Redis command elects one refresh winner and retains spent state to expiry."""
    remaining = math.ceil(payload["exp"] - time.time())
    if remaining <= 0:
        raise HTTPException(401, "Invalid refresh token")
    try:
        return bool(await r.set(_blocklist_key(token), "revoked", nx=True, ex=remaining))
    except RedisError:
        # A timeout might follow a successful SET. Never issue a replacement
        # when its result is unknown; the user may need to sign in again.
        raise HTTPException(503, "Authentication service temporarily unavailable") from None


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


@router.get("/capabilities", response_model=AuthCapabilities)
async def capabilities() -> AuthCapabilities:
    """Publish only the sign-up capability needed by the unauthenticated UI."""
    return AuthCapabilities(public_registration_enabled=settings.registration_enabled)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(_CREDENTIAL_RATE_LIMIT)
async def register(
    body: RegisterRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
    r: aioredis.Redis = Depends(get_redis),
) -> TokenResponse:
    """Create a viewer when public registration is enabled."""
    if not settings.registration_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public registration is disabled; contact an administrator for access",
        )
    if str(body.email).casefold() == settings.demo_email.casefold():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The demo account must be provisioned by an administrator",
        )
    await check_account_limit(r, str(body.email))
    existing = await session.exec(select(User).where(User.email == body.email))
    if existing.first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=body.email,
        full_name=body.full_name,
        hashed_password=await run_in_threadpool(hash_password, body.password),
        role=UserRole.viewer,
    )
    session.add(user)
    await session.flush()  # get the UUID before commit

    await _audit(session, request, user.id, "register")

    access = create_access_token(str(user.id), user.role.value)
    refresh = create_refresh_token(str(user.id), user.hashed_password)
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
    r: aioredis.Redis = Depends(get_redis),
) -> TokenResponse:
    """Authenticate and return tokens."""
    await check_account_limit(r, str(body.email))
    result = await session.exec(select(User).where(User.email == body.email))
    user = result.first()

    # Always pay for a bcrypt verify, even with no user to verify against —
    # `user is None or not verify_password(...)` would short-circuit, and the
    # ~180ms gap between "no such email" and "wrong password" is a readable
    # account-enumeration oracle over the network.
    if not await run_in_threadpool(
        verify_password_or_dummy, body.password, user.hashed_password if user else None
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    ensure_demo_is_viewer(user)

    await _audit(session, request, user.id, "login")

    access = create_access_token(str(user.id), user.role.value)
    refresh = create_refresh_token(str(user.id), user.hashed_password)
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

    subject = token_subject(payload)
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    user = await session.get(User, subject)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    ensure_demo_is_viewer(user)

    credential = payload.get("credential")
    if not isinstance(credential, str) or not hmac.compare_digest(
        credential, credential_fingerprint(user.hashed_password)
    ):
        raise HTTPException(401, "Session expired; sign in again")

    if not await _consume(r, refresh_token, payload):
        raise HTTPException(401, "Token revoked")

    new_access = create_access_token(str(user.id), user.role)
    new_refresh = create_refresh_token(str(user.id), user.hashed_password)

    _set_refresh_cookie(response, new_refresh)

    return TokenResponse(access_token=new_access)


# ── Logout ────────────────────────────────────────────────────────────────────


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
    r: aioredis.Redis = Depends(get_redis),
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
) -> None:
    """Revoke the refresh token and clear the cookie."""
    # Logout remains available after access expiry and account disabling.
    # Invalid/legacy cookies need only be cleared: refresh rejects them already.
    payload = decode_refresh_token(refresh_token) if refresh_token else None
    if payload is not None and payload.get("type") == "refresh":
        await _consume(r, refresh_token, payload)
        subject = token_subject(payload)
        if subject is not None and await session.get(User, subject) is not None:
            await _audit(session, request, subject, "logout")

    response.delete_cookie(
        key=REFRESH_COOKIE,
        path="/api/v1/auth",
        secure=settings.environment != "development",
        httponly=True,
        samesite="lax",
    )


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
