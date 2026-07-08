"""Role-based access-control matrix: every guard class x every role.

ResolveAI gates routes with three dependency guards (app.core.deps):

    get_current_user  -> any authenticated, active user
    require_writer    -> admin or analyst  (a viewer is read-only)
    require_admin     -> admin only

This suite pins the authorization *contract* at a representative endpoint for
each guard, sweeping all three roles plus the unauthenticated case. It asserts
the authz boundary (401 unauthenticated / 403 wrong-role / neither when the role
is allowed) rather than exact business status codes, so it stays green as the
handlers evolve. Auth is faked at get_current_user -- require_writer and
require_admin both mount on it, so the injected role flows straight through.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.user import UserRole

# One representative endpoint per guard class. call_kwargs go straight to
# httpx.request. The admin body is schema-valid but points outside the import
# sandbox, so an admin clears authz and stops at a 400 -- no real ingest fires.
_ENDPOINTS = [
    ("reader", "GET", "/api/v1/workspace/board", {}),
    ("reader", "GET", "/api/v1/complaints/", {}),
    ("writer", "POST", "/api/v1/workspace/enqueue/classification", {"params": {"limit": 1}}),
    ("admin", "POST", "/api/v1/complaints/bulk-import", {"json": {"path": "/tmp/nope.csv"}}),
]

# guard class -> role -> is this role authorized? (False => expect a 403)
_ALLOWED = {
    "reader": {UserRole.admin: True, UserRole.analyst: True, UserRole.viewer: True},
    "writer": {UserRole.admin: True, UserRole.analyst: True, UserRole.viewer: False},
    "admin": {UserRole.admin: True, UserRole.analyst: False, UserRole.viewer: False},
}

_ROLES = [UserRole.admin, UserRole.analyst, UserRole.viewer]


def _user(role: UserRole) -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), email=f"{role.value}@test.com", role=role)


@pytest_asyncio.fixture()
async def factory():
    import app.models  # noqa: F401 -- registers tables on SQLModel.metadata
    from app.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture()
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _build_app(factory, fake_redis, *, role: UserRole | None):
    """App on sqlite + fakeredis. A role fakes the current user; role=None leaves
    auth real so the request exercises the genuine 401 (no-token) path."""
    from app.core.deps import get_current_user, get_redis
    from app.database import get_session
    from app.main import create_app

    async def override_session():
        async with factory() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    app = create_app()
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_redis] = lambda: fake_redis
    if role is not None:
        app.dependency_overrides[get_current_user] = lambda: _user(role)
    return app


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
@pytest.mark.parametrize("guard, method, url, call_kwargs", _ENDPOINTS)
@pytest.mark.parametrize("role", _ROLES)
async def test_role_endpoint_matrix(factory, fake_redis, role, guard, method, url, call_kwargs):
    app = _build_app(factory, fake_redis, role=role)
    async with _client(app) as ac:
        r = await ac.request(method, url, **call_kwargs)
    if _ALLOWED[guard][role]:
        # Authorized: authn + authz both cleared. The exact code is the handler's
        # business (200 board/list/enqueue, 400 out-of-sandbox import) -- all we
        # assert here is that neither auth gate rejected the request.
        assert r.status_code not in (401, 403), (role.value, guard, url, r.status_code)
    else:
        assert r.status_code == 403, (role.value, guard, url, r.status_code)


@pytest.mark.asyncio
@pytest.mark.parametrize("guard, method, url, call_kwargs", _ENDPOINTS)
async def test_unauthenticated_is_401(factory, fake_redis, guard, method, url, call_kwargs):
    app = _build_app(factory, fake_redis, role=None)  # real auth, no bearer token
    async with _client(app) as ac:
        r = await ac.request(method, url, **call_kwargs)
    assert r.status_code == 401, (guard, url, r.status_code)
