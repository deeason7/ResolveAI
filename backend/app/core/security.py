"""
JWT creation/verification and bcrypt password hashing.

Two tokens are issued on login:
  - access token  (short-lived, 30 min) — sent in Authorization header
  - refresh token (long-lived, 7 days)  — stored in httpOnly cookie

The refresh token lets the client get a new access token without re-logging-in.
Revoked refresh tokens are tracked in Redis (blocklist pattern).
"""

import hashlib
import hmac
import logging
import secrets
import uuid
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


# A hash of a random string nobody holds the password for. Login verifies
# against this when the email doesn't exist, so a miss costs the same ~180ms of
# bcrypt as a hit — otherwise "unknown email" returns instantly and the response
# time tells an attacker which addresses are registered.
_ABSENT_USER_HASH = hash_password(secrets.token_urlsafe(32))


def verify_password_or_dummy(plain: str, hashed: str | None) -> bool:
    """Verify a password, burning the same work when there is no user to verify.

    Args:
        plain: The submitted password.
        hashed: The stored bcrypt hash, or None when no such account exists.

    Returns:
        True only if hashed is present and matches. A None hash always returns
        False — after doing the bcrypt work anyway.
    """
    if hashed is None:
        bcrypt.checkpw(plain.encode(), _ABSENT_USER_HASH.encode())
        return False
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
    """Keyed one-way hash of an IP address, for audit logs.

    HMAC rather than a bare digest: IPv4 is only 2^32 values, so a plain
    sha256(ip) is reversible by anyone willing to spend a few seconds building
    the table. Keying it with a secret the attacker doesn't have is what makes
    the hash actually one-way for someone holding a dump of audit_logs.
    """
    return hmac.new(settings.jwt_secret_key.encode(), ip.encode(), hashlib.sha256).hexdigest()


def token_subject(payload: dict) -> uuid.UUID | None:
    """Parse the `sub` claim as a UUID, or None if it isn't one.

    The claim is only ever written by us, so a bad value means something is
    already wrong — but bare uuid.UUID() raises ValueError, which surfaces as a
    500. A malformed token is a 401, not a server error.
    """
    try:
        return uuid.UUID(str(payload.get("sub")))
    except (ValueError, AttributeError, TypeError):
        return None


def hash_token(token: str) -> str:
    """Digest a token for use as a cache/blocklist key.

    Keeps raw refresh JWTs out of Redis — a dump of the blocklist then leaks
    nothing usable. Unkeyed sha256 is enough here: a JWT has far too much
    entropy to enumerate, unlike an IP address.
    """
    return hashlib.sha256(token.encode()).hexdigest()
