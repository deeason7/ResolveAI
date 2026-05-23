"""
JWT creation/verification and bcrypt password hashing.

Two tokens are issued on login:
  - access token  (short-lived, 30 min) — sent in Authorization header
  - refresh token (long-lived, 7 days)  — stored in httpOnly cookie

The refresh token lets the client get a new access token without re-logging-in.
Revoked refresh tokens are tracked in Redis (blocklist pattern).
"""

import hashlib
import logging
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.config import settings

logger = logging.getLogger(__name__)

# ── Password hashing ──────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt (cost factor 12)."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT helpers ───────────────────────────────────────────────────────────────

TokenPayload = dict[str, str | int | datetime]


def _make_token(payload: TokenPayload, secret: str, expires_delta: timedelta) -> str:
    expire = datetime.now(UTC) + expires_delta
    return jwt.encode({**payload, "exp": expire}, secret, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str, role: str) -> str:
    """Create a short-lived access JWT."""
    return _make_token(
        {"sub": user_id, "role": role, "type": "access"},
        settings.jwt_secret_key,
        timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(user_id: str) -> str:
    """Create a long-lived refresh JWT."""
    return _make_token(
        {"sub": user_id, "type": "refresh"},
        settings.jwt_refresh_secret_key,
        timedelta(days=settings.refresh_token_expire_days),
    )


def decode_access_token(token: str) -> dict | None:
    """Decode and validate an access token. Returns None on any failure."""
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        logger.debug("Access token expired")
        return None
    except jwt.InvalidTokenError:
        logger.debug("Invalid access token")
        return None


def decode_refresh_token(token: str) -> dict | None:
    """Decode and validate a refresh token. Returns None on any failure."""
    try:
        return jwt.decode(
            token, settings.jwt_refresh_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.InvalidTokenError:
        return None


# ── Utilities ─────────────────────────────────────────────────────────────────


def hash_ip(ip: str) -> str:
    """One-way hash an IP address for audit logs (privacy-preserving)."""
    return hashlib.sha256(ip.encode()).hexdigest()
