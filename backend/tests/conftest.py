"""
Shared test fixtures.

Env vars must be set at module level — before any app import — so that
pydantic-settings picks them up when Settings() is first instantiated.

We use a file-based SQLite DB (deleted after each test) instead of :memory:
because in-memory DBs combined with aiosqlite + StaticPool are fragile across
event loops. A tmp file is reliable and only marginally slower.
"""

import os
import tempfile
from pathlib import Path

# Generate a unique sqlite file per test process
_DB_PATH = Path(tempfile.gettempdir()) / f"resolveai_test_{os.getpid()}.db"
TEST_DATABASE_URL = f"sqlite+aiosqlite:///{_DB_PATH}"

os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-that-is-long-enough-32chars")
os.environ.setdefault("JWT_REFRESH_SECRET_KEY", "test-refresh-key-that-is-long-enough-32c")
os.environ.setdefault("ENVIRONMENT", "test")

import fakeredis
import fakeredis.aioredis
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from unittest.mock import patch

_fake_redis_server = fakeredis.FakeServer()


def _make_fake_redis(*args, **kwargs):
    return fakeredis.aioredis.FakeRedis(server=_fake_redis_server, decode_responses=True)


@pytest_asyncio.fixture()
async def client():
    """Per-test: reset schema, return an HTTP client."""
    # Import models FIRST so SQLModel.metadata knows about all tables
    import app.models  # noqa: F401
    from app.database import engine, get_session

    # Fresh schema for this test — wipe and recreate tables on the live engine
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def override_session():
        async with factory() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    from app.main import create_app
    test_app = create_app()
    test_app.dependency_overrides[get_session] = override_session

    with patch("app.api.routes.auth.aioredis.from_url", side_effect=_make_fake_redis):
        async with AsyncClient(
            transport=ASGITransport(app=test_app), base_url="http://test"
        ) as ac:
            yield ac
    # Do NOT dispose the module-level engine — it's reused across tests.
    # The next test's setup wipes the schema via drop_all/create_all.
