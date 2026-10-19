"""Safety boundary for the disposable PostgreSQL test harness."""

from __future__ import annotations

import re
from collections.abc import Mapping

from sqlalchemy.engine import URL, make_url


def test_database_target(environment: Mapping[str, str]) -> URL:
    """Validate the explicit target before connecting or resetting its schema.

    A loopback-only URL prevents accidentally using a managed database. Query
    parameters are forbidden because asyncpg accepts connection overrides there.
    The opt-in permits deletion of every object in this database's public schema.
    """
    if environment.get("TEST_DATABASE_ALLOW_RESET") != "yes":
        raise ValueError("Set TEST_DATABASE_ALLOW_RESET=yes for a disposable test database")
    raw_url = environment.get("TEST_DATABASE_URL")
    if not raw_url:
        raise ValueError("TEST_DATABASE_URL is required; DATABASE_URL is never a fallback")
    try:
        url = make_url(raw_url)
    except Exception:
        raise ValueError("TEST_DATABASE_URL is not a valid SQLAlchemy URL") from None
    if url.drivername != "postgresql+asyncpg":
        raise ValueError("TEST_DATABASE_URL must use postgresql+asyncpg")
    if url.host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("TEST_DATABASE_URL must use a loopback host")
    if not url.database or not re.fullmatch(r"resolveai_test(?:_[a-z0-9]+)*", url.database):
        raise ValueError("Test database name must be resolveai_test or resolveai_test_<suffix>")
    if not url.username or url.query:
        raise ValueError("Test URL needs an explicit user and must not contain query parameters")
    return url


def verify_connected_database(actual: str, expected: str) -> None:
    """Check the connected identity before any destructive statement."""
    if actual != expected:
        raise ValueError("Connected database does not match the validated test target")
