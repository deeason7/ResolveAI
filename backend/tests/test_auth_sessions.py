"""Session replay, account throttling and bcrypt boundaries with isolated I/O."""

import asyncio
import threading
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from redis.exceptions import AuthenticationError, OutOfMemoryError, ResponseError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlmodel import select

from app.config import settings
from app.core.deps import get_redis
from app.core.security import (
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    verify_password,
    verify_password_or_dummy,
)
from app.database import AsyncSessionLocal
from app.models.user import User, UserRole
from app.services.auth_rate_limit import ACCOUNT_ATTEMPTS_PER_MINUTE, account_key

REGISTER = "/api/v1/auth/register"
LOGIN = "/api/v1/auth/login"
REFRESH = "/api/v1/auth/refresh"
LOGOUT = "/api/v1/auth/logout"
BODY = {
    "email": "sessions@example.com",
    "full_name": "Synthetic User",
    "password": "synthetic-password",
}


def cookie(token):
    return {"Cookie": f"refresh_token={token}"}


async def register(client):
    response = await client.post(REGISTER, json=BODY)
    assert response.status_code == 201
    return response


def test_tokens_issued_in_the_same_second_have_independent_ids():
    instant = datetime.now(UTC)
    with patch("app.core.security.datetime") as clock:
        clock.now.return_value = instant
        # Same subject, credentials and expiry still produce distinct tokens.
        tokens = [
            create_refresh_token("00000000-0000-0000-0000-000000000001", "hash") for _ in range(10)
        ]
    payloads = [decode_refresh_token(token) for token in tokens]
    assert len({p["exp"] for p in payloads}) == 1
    assert len({p["jti"] for p in payloads}) == len(tokens)


async def test_concurrent_refresh_has_exactly_one_winner(client, fake_redis):
    initial = await register(client)
    token = initial.cookies["refresh_token"]
    original_set = fake_redis.set
    ready = asyncio.Event()
    count = 0
    contenders = 12

    async def synchronized_set(*args, **kwargs):
        nonlocal count
        count += 1
        if count == contenders:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=5)
        return await original_set(*args, **kwargs)

    fake_redis.set = synchronized_set
    client._transport.app.dependency_overrides[get_redis] = lambda: fake_redis
    results = await asyncio.gather(
        *(client.post(REFRESH, headers=cookie(token)) for _ in range(contenders))
    )
    assert [r.status_code for r in results].count(200) == 1
    assert [r.status_code for r in results].count(401) == contenders - 1
    winner = next(r for r in results if r.status_code == 200)
    assert winner.cookies["refresh_token"] != token
    fake_redis.set = original_set
    assert (await client.post(REFRESH, headers=cookie(token))).status_code == 401
    assert (
        await client.post(REFRESH, headers=cookie(winner.cookies["refresh_token"]))
    ).status_code == 200


async def test_logout_invalidates_only_its_session_without_access_token(client):
    first = await register(client)
    second = await client.post(LOGIN, json={"email": BODY["email"], "password": BODY["password"]})
    first_token, second_token = first.cookies["refresh_token"], second.cookies["refresh_token"]
    assert first_token != second_token
    logout = await client.post(LOGOUT, headers=cookie(first_token))
    assert logout.status_code == 204
    assert "Max-Age=0" in logout.headers["set-cookie"]
    assert (await client.post(REFRESH, headers=cookie(first_token))).status_code == 401
    assert (await client.post(REFRESH, headers=cookie(second_token))).status_code == 200
    assert (await client.post(LOGOUT, headers=cookie(first_token))).status_code == 204


async def test_spent_token_ttl_matches_remaining_lifetime(client, fake_redis):
    from app.api.routes.auth import _blocklist_key

    response = await register(client)
    payload = decode_refresh_token(response.cookies["refresh_token"])
    payload["exp"] = int(time.time()) + 31
    short = jwt.encode(payload, settings.jwt_refresh_secret_key, algorithm=settings.jwt_algorithm)
    assert (await client.post(REFRESH, headers=cookie(short))).status_code == 200
    assert 1 <= await fake_redis.ttl(_blocklist_key(short)) <= 31
    assert short not in " ".join(await fake_redis.keys("*"))


@pytest.mark.parametrize(
    "claim,value",
    [("exp", "9999999999"), ("credential", "é" * 64), ("jti", ""), ("credential", None)],
)
async def test_signed_but_malformed_refresh_is_rejected_without_server_error(client, claim, value):
    response = await register(client)
    payload = decode_refresh_token(response.cookies["refresh_token"])
    payload[claim] = value
    token = jwt.encode(payload, settings.jwt_refresh_secret_key, algorithm=settings.jwt_algorithm)
    assert (await client.post(REFRESH, headers=cookie(token))).status_code == 401


@pytest.mark.parametrize("path", [REFRESH, LOGOUT])
async def test_redis_outage_never_issues_replacement_or_claims_logout(client, fake_redis, path):
    initial = await register(client)
    fake_redis.set = AsyncMock(side_effect=RedisConnectionError("synthetic outage"))
    client._transport.app.dependency_overrides[get_redis] = lambda: fake_redis
    response = await client.post(path, headers=cookie(initial.cookies["refresh_token"]))
    assert response.status_code == 503
    assert "access_token" not in response.json()
    assert "set-cookie" not in response.headers


async def test_refresh_reads_current_role_and_disabled_state(client):
    initial = await register(client)
    token = initial.cookies["refresh_token"]
    async with AsyncSessionLocal() as session, session.begin():
        user = (await session.exec(select(User))).one()
        user.role = UserRole.analyst
        session.add(user)
    response = await client.post(REFRESH, headers=cookie(token))
    assert response.status_code == 200
    profile = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {response.json()['access_token']}"}
    )
    assert profile.json()["role"] == "analyst"
    async with AsyncSessionLocal() as session, session.begin():
        user = (await session.exec(select(User))).one()
        user.is_active = False
        session.add(user)
    latest = response.cookies["refresh_token"]
    assert (await client.post(REFRESH, headers=cookie(latest))).status_code == 401
    assert (await client.post(LOGOUT, headers=cookie(latest))).status_code == 204


@pytest.mark.parametrize("password", ["a" * 73, "é" * 37, "雪" * 25])
def test_hash_and_verification_never_truncate_utf8(password):
    with patch("app.core.security.bcrypt.checkpw") as checker:
        with pytest.raises(ValueError, match="72 UTF-8 bytes"):
            verify_password(password, "unused")
        with pytest.raises(ValueError, match="72 UTF-8 bytes"):
            verify_password_or_dummy(password, None)
        checker.assert_not_called()
    with patch("app.core.security.bcrypt.hashpw") as hasher:
        with pytest.raises(ValueError, match="72 UTF-8 bytes"):
            hash_password(password)
        hasher.assert_not_called()


@pytest.mark.parametrize("password", ["a" * 72, "é" * 36, "雪" * 24])
def test_exact_72_byte_password_verifies_but_any_suffix_is_rejected(password):
    hashed = hash_password(password)
    assert verify_password(password, hashed)
    for suffix in ("a", "b"):
        with pytest.raises(ValueError, match="72 UTF-8 bytes"):
            verify_password(password + suffix, hashed)


@pytest.mark.parametrize("path", [REGISTER, LOGIN])
@pytest.mark.parametrize("password", ["a" * 73, "é" * 37])
async def test_api_byte_boundary_returns_safe_validation_error(client, path, password):
    response = await client.post(path, json={**BODY, "password": password})
    assert response.status_code == 422
    assert password not in response.text
    assert "72 UTF-8 bytes" in response.text
    assert "input" not in response.json()["detail"][0]


async def test_account_ceiling_is_shared_without_storing_email_or_trusting_forwarded_ip(
    client, fake_redis
):
    with patch("app.api.routes.auth.verify_password_or_dummy", return_value=False) as verify:
        for i in range(ACCOUNT_ATTEMPTS_PER_MINUTE):
            response = await client.post(
                LOGIN, json=BODY, headers={"X-Forwarded-For": f"198.51.100.{i}"}
            )
            assert response.status_code == 401
        response = await client.post(LOGIN, json=BODY, headers={"X-Forwarded-For": "203.0.113.99"})
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert verify.call_count == ACCOUNT_ATTEMPTS_PER_MINUTE
        # Sharing Streamlit's peer address doesn't share the account allowance.
        assert (
            await client.post(LOGIN, json={**BODY, "email": "other@example.com"})
        ).status_code == 401
    keys = await fake_redis.keys("limit:credentials:*")
    assert keys
    for key in keys:
        assert "@" not in key
        assert 1 <= await fake_redis.ttl(key) <= 120
    assert account_key("CASE@example.com", 1) == account_key("case@example.com", 1)


@pytest.mark.parametrize(
    "path,function", [(LOGIN, "verify_password_or_dummy"), (REGISTER, "hash_password")]
)
async def test_bcrypt_work_does_not_block_health_handler(client, monkeypatch, path, function):
    started, release = threading.Event(), threading.Event()
    event_loop_thread = threading.get_ident()

    def blocking_work(*_args):
        assert threading.get_ident() != event_loop_thread
        started.set()
        assert release.wait(timeout=5)
        return False if function == "verify_password_or_dummy" else "synthetic-hash"

    monkeypatch.setattr(f"app.api.routes.auth.{function}", blocking_work)
    task = asyncio.create_task(client.post(path, json=BODY))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        health = await asyncio.wait_for(client.get("/api/v1/health"), timeout=1)
        assert health.status_code == 200
    finally:
        release.set()
        result = await task
    assert result.status_code == (401 if path == LOGIN else 201)


@pytest.mark.parametrize("path", [REGISTER, LOGIN])
async def test_credentials_fail_closed_when_throttle_store_is_unavailable(client, fake_redis, path):
    pipeline = fake_redis.pipeline()
    pipeline.execute = AsyncMock(side_effect=RedisConnectionError("synthetic outage"))
    fake_redis.pipeline = lambda **_kwargs: pipeline
    client._transport.app.dependency_overrides[get_redis] = lambda: fake_redis
    response = await client.post(path, json=BODY)
    assert response.status_code == 503
    assert "set-cookie" not in response.headers
    async with AsyncSessionLocal() as session:
        assert (await session.exec(select(User))).all() == []


@pytest.mark.parametrize("path", [LOGIN, REFRESH])
@pytest.mark.parametrize(
    "error",
    [RedisConnectionError, RedisTimeoutError, AuthenticationError, OutOfMemoryError, ResponseError],
)
async def test_aggregate_limiter_outage_is_an_explicit_503(client, monkeypatch, path, error):
    from app.middleware.rate_limit import limiter

    def unavailable(*_args, **_kwargs):
        raise error("synthetic endpoint limit store outage")

    monkeypatch.setattr(limiter._limiter, "hit", unavailable)
    response = await client.post(path, json=BODY)
    assert response.status_code == 503
    assert "set-cookie" not in response.headers
    assert "synthetic endpoint limit" not in response.text
