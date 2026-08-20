"""
FastAPI dependencies for authentication and database access.

Any route that needs the current user just declares:
    current_user: User = Depends(get_current_user)

FastAPI injects it automatically — the route itself never touches JWT logic.
"""

import logging
from functools import lru_cache

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.core.security import decode_access_token, token_subject
from app.database import get_session
from app.models.user import User, UserRole
from app.services.graph_store import GraphStore, get_default_graph_store

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_session),
) -> User:
    """Extract and validate the Bearer token; return the matching User row.

    Raises:
        HTTPException 401: if token is missing, expired, or invalid.
        HTTPException 403: if the user account is inactive.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    subject = token_subject(payload)
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await session.get(User, subject)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Gate a route to admin users only. Mounts on top of get_current_user."""
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return current_user


# Roles allowed to change state. An allow-list (not "role != viewer") is the
# point: a role we add later is read-only until it's explicitly granted write —
# authz should fail closed, not open.
_WRITER_ROLES = frozenset({UserRole.admin, UserRole.analyst})


def require_writer(current_user: User = Depends(get_current_user)) -> User:
    """Gate a route to roles that may mutate state (analyst, admin).

    A viewer authenticates normally and can read every page, but can't submit,
    enqueue, generate, approve, or reject. Mounts on get_current_user and returns
    the same User, so it's a drop-in for it at any write call site.
    """
    if current_user.role not in _WRITER_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires a writer role (analyst or admin)",
        )
    return current_user


def get_graph_store() -> GraphStore:
    """Inject the process-wide Neo4j graph store.

    A thin Depends() wrapper over the lru_cache singleton so routes stay
    decoupled from how the store is built — and tests can swap a mock via
    app.dependency_overrides[get_graph_store] without monkeypatching the module.
    """
    return get_default_graph_store()


@lru_cache
def get_default_redis() -> aioredis.Redis:
    """Process-wide Redis client (connection pool included).

    Routes only produce to streams (XADD), so one shared client is plenty.
    Lives here rather than a service module because there's no domain logic to
    wrap — it IS just the client. Closed by the app's lifespan on shutdown.
    """
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def get_redis() -> aioredis.Redis:
    """Depends() seam over the Redis singleton; override in tests with fakeredis."""
    return get_default_redis()
