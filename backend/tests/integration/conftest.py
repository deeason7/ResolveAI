"""Independent PostgreSQL engines; never use the application's SQLite singleton."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel.ext.asyncio.session import AsyncSession

from tests.postgres_support import test_database_target as database_target
from tests.postgres_support import verify_connected_database

BACKEND = Path(__file__).resolve().parents[2]


@dataclass
class PostgresDatabase:
    engine: AsyncEngine
    url: URL
    workdir: Path

    async def reset(self) -> None:
        async with self.engine.begin() as connection:
            actual = await connection.scalar(text("SELECT current_database()"))
            verify_connected_database(actual, self.url.database)
            await connection.execute(text("DROP SCHEMA public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))

    async def upgrade(self, revision: str = "head") -> None:
        # Alembic's env.py runs its own event loop. A subprocess also isolates
        # DATABASE_URL from the unit suite and cwd from any developer .env file.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(BACKEND),
            "DATABASE_URL": self.url.render_as_string(hide_password=False),
            "DB_REQUIRE_SSL": "false",
            "ENVIRONMENT": "test",
            "REDIS_URL": "redis://127.0.0.1:1/0",
            "NEO4J_URI": "bolt://127.0.0.1:1",
            "NEO4J_PASSWORD": "test",
            "JWT_SECRET_KEY": "test-secret-key-that-is-long-enough-32chars",
            "JWT_REFRESH_SECRET_KEY": "test-refresh-key-that-is-long-enough-32c",
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(BACKEND / "alembic.ini"),
            "upgrade",
            revision,
            cwd=self.workdir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=60)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            details = output.decode(errors="replace")
            details = details.replace(env["DATABASE_URL"], "[test database]")
            if self.url.password:
                details = details.replace(self.url.password, "[redacted]")
            pytest.fail(f"Alembic upgrade failed:\n{details}")


@pytest.fixture(scope="session")
def postgres_target(pytestconfig):
    if not pytestconfig.getoption("--run-postgres"):
        pytest.skip("PostgreSQL integration tests require --run-postgres")
    try:
        return database_target(os.environ)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from None


@pytest_asyncio.fixture()
async def postgres_database(postgres_target, tmp_path):
    engine = create_async_engine(
        postgres_target,
        poolclass=NullPool,
        connect_args={"timeout": 5, "command_timeout": 15},
    )
    database = PostgresDatabase(engine, postgres_target, tmp_path)
    try:
        await database.reset()
        yield database
    finally:
        await engine.dispose()


@pytest_asyncio.fixture()
async def pg_engine(postgres_database):
    await postgres_database.upgrade()
    yield postgres_database.engine


@pytest.fixture()
def pg_sessionmaker(pg_engine):
    return async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)
