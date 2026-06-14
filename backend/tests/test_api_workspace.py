"""Tests for the Workspace pipeline API.

SQLite via the session override, fakeredis behind get_redis, auth bypassed —
these exercise the stage-count board, the two batch-enqueue routes (pending ->
classification stream; escalated -> resolution stream with the agent_triggered
flip), the per-call limit cap, and the 401 path. No worker or LLM involved.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.models.complaint import Complaint, ComplaintStatus

BOARD_URL = "/api/v1/workspace/board"
CLASSIFY_URL = "/api/v1/workspace/enqueue/classification"
RESOLVE_URL = "/api/v1/workspace/enqueue/resolution"

_DUMMY_USER = SimpleNamespace(id=uuid.uuid4(), email="analyst@test.com")


@pytest_asyncio.fixture()
async def factory():
    import app.models  # noqa: F401 — registers tables on SQLModel.metadata
    from app.database import engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    yield async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture()
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _build_app(factory, fake_redis, *, bypass_auth=True):
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
    if bypass_auth:
        app.dependency_overrides[get_current_user] = lambda: _DUMMY_USER
    return app


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _seed(factory, **status_counts: int) -> None:
    """Insert complaints in the given statuses (keyword = ComplaintStatus name)."""
    async with factory() as s:
        for name, count in status_counts.items():
            status = ComplaintStatus[name]
            for _ in range(count):
                s.add(Complaint(narrative="x", status=status))
        await s.commit()


@pytest.mark.asyncio
async def test_board_counts_by_status(factory, fake_redis):
    await _seed(factory, pending=5, classified=2, escalated=3, resolved=1)
    app = _build_app(factory, fake_redis)
    async with _client(app) as ac:
        r = await ac.get(BOARD_URL)
    assert r.status_code == 200
    body = r.json()
    assert body["pending"] == 5
    assert body["classified"] == 2
    assert body["escalated"] == 3
    assert body["resolved"] == 1
    assert body["agent_triggered"] == 0
    assert body["total"] == 11
    # Stream telemetry is present and zeroed — no worker/groups in the fake.
    assert body["classification_stream"]["name"] == settings.classification_queue
    assert body["classification_stream"]["in_flight"] == 0
    assert body["resolution_stream"]["name"] == settings.resolution_queue


@pytest.mark.asyncio
async def test_enqueue_classification_pushes_only_pending(factory, fake_redis):
    await _seed(factory, pending=4, classified=3)  # classified must NOT be enqueued
    app = _build_app(factory, fake_redis)
    async with _client(app) as ac:
        r = await ac.post(CLASSIFY_URL, params={"limit": 10})
    assert r.status_code == 200
    assert r.json()["enqueued"] == 4
    assert await fake_redis.xlen(settings.classification_queue) == 4


@pytest.mark.asyncio
async def test_enqueue_classification_respects_limit(factory, fake_redis):
    await _seed(factory, pending=10)
    app = _build_app(factory, fake_redis)
    async with _client(app) as ac:
        r = await ac.post(CLASSIFY_URL, params={"limit": 3})
    assert r.json()["enqueued"] == 3
    assert await fake_redis.xlen(settings.classification_queue) == 3


@pytest.mark.asyncio
async def test_enqueue_resolution_flips_escalated(factory, fake_redis):
    await _seed(factory, escalated=2, classified=5)  # only escalated are eligible
    app = _build_app(factory, fake_redis)
    async with _client(app) as ac:
        r = await ac.post(RESOLVE_URL, params={"limit": 10})
    assert r.status_code == 200
    assert r.json()["enqueued"] == 2
    assert await fake_redis.xlen(settings.resolution_queue) == 2
    async with factory() as s:
        statuses = [str(v) for v in (await s.exec(select(Complaint.status))).all()]
    assert statuses.count(ComplaintStatus.agent_triggered.value) == 2
    assert statuses.count(ComplaintStatus.classified.value) == 5


@pytest.mark.asyncio
async def test_enqueue_limit_cap_rejected(factory, fake_redis):
    app = _build_app(factory, fake_redis)
    async with _client(app) as ac:
        r = await ac.post(CLASSIFY_URL, params={"limit": 9999})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_board_requires_auth(factory, fake_redis):
    app = _build_app(factory, fake_redis, bypass_auth=False)
    async with _client(app) as ac:
        r = await ac.get(BOARD_URL)
    assert r.status_code == 401
