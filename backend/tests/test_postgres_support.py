"""The destructive-test boundary can be checked without a running database."""

import pytest

from tests.postgres_support import test_database_target as database_target
from tests.postgres_support import verify_connected_database


def _env(url="postgresql+asyncpg://test:test@127.0.0.1:55432/resolveai_test"):
    return {"TEST_DATABASE_URL": url, "TEST_DATABASE_ALLOW_RESET": "yes"}


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_accepts_explicit_disposable_loopback_target(host):
    target = database_target(_env(f"postgresql+asyncpg://test:test@{host}/resolveai_test_ci"))
    assert target.database == "resolveai_test_ci"


def test_requires_explicit_reset_opt_in():
    with pytest.raises(ValueError, match="ALLOW_RESET"):
        database_target({"TEST_DATABASE_URL": _env()["TEST_DATABASE_URL"]})


def test_never_falls_back_to_application_database_url():
    with pytest.raises(ValueError, match="never a fallback"):
        database_target(
            {"DATABASE_URL": _env()["TEST_DATABASE_URL"], "TEST_DATABASE_ALLOW_RESET": "yes"}
        )


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://test:test@database.example/resolveai_test",
        "postgresql+asyncpg://test:test@127.0.0.1/resolveai_db",
        "postgresql+asyncpg://test:test@127.0.0.1/postgres",
        "postgresql+asyncpg://test:test@127.0.0.1/resolveai_test?host=database.example",
        "postgresql+asyncpg:///resolveai_test",
        "sqlite+aiosqlite:///resolveai_test",
        "not-a-url-with-a-private-password",
    ],
)
def test_refuses_unsafe_targets_without_echoing_credentials(url):
    with pytest.raises(ValueError) as caught:
        database_target(_env(url))
    assert url not in str(caught.value)


def test_connected_identity_must_match_before_reset():
    with pytest.raises(ValueError, match="does not match"):
        verify_connected_database("resolveai_db", "resolveai_test")
    verify_connected_database("resolveai_test", "resolveai_test")
