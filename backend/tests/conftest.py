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
UNIT_DATABASE_URL = f"sqlite+aiosqlite:///{_DB_PATH}"

# The app singleton always belongs to the unit suite. PostgreSQL fixtures use
# TEST_DATABASE_URL directly and never reuse this engine or DATABASE_URL.
# Never inherit a developer's production service settings or provider keys.
# TEST_DATABASE_URL is deliberately left alone for the guarded PG fixtures.
os.environ.update(
    DATABASE_URL=UNIT_DATABASE_URL,
    DB_REQUIRE_SSL="false",
    REDIS_URL="redis://127.0.0.1:1/1",
    NEO4J_URI="bolt://127.0.0.1:1",
    NEO4J_PASSWORD="test",
    QDRANT_URL="",
    QDRANT_HOST="127.0.0.1",
    QDRANT_PORT="1",
    QDRANT_API_KEY="",
    OLLAMA_BASE_URL="http://127.0.0.1:1",
    GROQ_API_KEY="",
    OPENAI_API_KEY="",
    JWT_SECRET_KEY="test-secret-key-that-is-long-enough-32chars",
    JWT_REFRESH_SECRET_KEY="test-refresh-key-that-is-long-enough-32c",
    ENVIRONMENT="test",
    PUBLIC_REGISTRATION_ENABLED="true",
    DEMO_EMAIL="demo@resolveai-demo.com",
)

import fakeredis
import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.database import engine as _unit_engine


@event.listens_for(_unit_engine.sync_engine, "connect")
def _enable_sqlite_foreign_keys(connection, _record):
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def pytest_addoption(parser):
    parser.addoption(
        "--run-postgres",
        action="store_true",
        help="Run destructive integration tests against an explicitly configured test database.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-postgres"):
        skip = pytest.mark.skip(reason="PostgreSQL integration tests require --run-postgres")
        for item in items:
            if "postgres" in item.keywords:
                item.add_marker(skip)


_fake_redis_server = fakeredis.FakeServer()


def _make_fake_redis(*args, **kwargs):
    return fakeredis.aioredis.FakeRedis(server=_fake_redis_server, decode_responses=True)


@pytest.fixture()
def fake_redis():
    """The same in-memory Redis the app's get_redis override hands out.

    All fakes share one FakeServer, so state written here is visible to the
    routes — which is what lets a test seed a key the endpoint should react to.
    """
    return _make_fake_redis()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Isolate rate-limit state between tests.

    The limiter is a module singleton and its dev/test backend is in-process
    memory, so counts persist across the whole session. Without this reset a
    long suite would eventually exhaust the global 200/min limit (every routed
    request shares the test client's address as the key), and per-route limit
    tests couldn't assume a clean window. Resetting the memory backend before
    each test makes both deterministic.
    """
    from app.middleware.rate_limit import limiter

    limiter._storage.reset()
    yield


@pytest_asyncio.fixture()
async def client():
    """Per-test: reset schema, return an HTTP client."""
    await _make_fake_redis().flushall()
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

    from app.core.deps import get_redis
    from app.main import create_app

    test_app = create_app()
    test_app.dependency_overrides[get_session] = override_session
    # Every Redis caller — stream producers and the auth refresh blocklist —
    # comes through this one seam now. Zero-arg
    # lambda on purpose: FastAPI introspects the override's signature, and a
    # (*args, **kwargs) factory would become two required query params.
    test_app.dependency_overrides[get_redis] = lambda: _make_fake_redis()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        yield ac
    # Do NOT dispose the module-level engine — it's reused across tests.
    # The next test's setup wipes the schema via drop_all/create_all.
