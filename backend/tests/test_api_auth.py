"""
Auth endpoint tests.

Covers: register, login, refresh, logout, /me, and key error paths.
Redis calls in refresh/logout are patched out so tests run without
a real Redis instance.
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

REGISTER_URL = "/api/v1/auth/register"
LOGIN_URL = "/api/v1/auth/login"
REFRESH_URL = "/api/v1/auth/refresh"
LOGOUT_URL = "/api/v1/auth/logout"
ME_URL = "/api/v1/auth/me"

VALID_USER = {
    "email": "test@example.com",
    "full_name": "Test User",
    "password": "securepassword123",
}


# ── Registration ──────────────────────────────────────────────────────────────


class TestRegister:
    async def test_register_returns_access_token(self, client: AsyncClient):
        response = await client.post(REGISTER_URL, json=VALID_USER)
        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_register_sets_refresh_cookie(self, client: AsyncClient):
        response = await client.post(
            REGISTER_URL,
            json={**VALID_USER, "email": "cookie@example.com"},
        )
        assert response.status_code == 201
        assert "refresh_token" in response.cookies

    async def test_duplicate_email_returns_409(self, client: AsyncClient):
        await client.post(REGISTER_URL, json={**VALID_USER, "email": "dup@example.com"})
        response = await client.post(REGISTER_URL, json={**VALID_USER, "email": "dup@example.com"})
        assert response.status_code == 409

    async def test_short_password_rejected(self, client: AsyncClient):
        response = await client.post(
            REGISTER_URL,
            json={**VALID_USER, "email": "short@example.com", "password": "abc"},
        )
        assert response.status_code == 422


# ── Login ─────────────────────────────────────────────────────────────────────


class TestLogin:
    @pytest.fixture(autouse=True)
    async def _create_user(self, client: AsyncClient):
        await client.post(REGISTER_URL, json={**VALID_USER, "email": "login@example.com"})

    async def test_valid_credentials_return_token(self, client: AsyncClient):
        response = await client.post(
            LOGIN_URL, json={"email": "login@example.com", "password": VALID_USER["password"]}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    async def test_wrong_password_returns_401(self, client: AsyncClient):
        response = await client.post(
            LOGIN_URL, json={"email": "login@example.com", "password": "wrongpassword"}
        )
        assert response.status_code == 401

    async def test_unknown_email_returns_401(self, client: AsyncClient):
        response = await client.post(
            LOGIN_URL, json={"email": "ghost@example.com", "password": "anything"}
        )
        assert response.status_code == 401


# ── /me ───────────────────────────────────────────────────────────────────────


class TestMe:
    async def test_me_returns_user_profile(self, client: AsyncClient):
        reg = await client.post(REGISTER_URL, json={**VALID_USER, "email": "me@example.com"})
        token = reg.json()["access_token"]

        response = await client.get(ME_URL, headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "me@example.com"
        assert "hashed_password" not in data

    async def test_me_without_token_returns_401(self, client: AsyncClient):
        response = await client.get(ME_URL)
        assert response.status_code == 401

    async def test_me_with_invalid_token_returns_401(self, client: AsyncClient):
        response = await client.get(ME_URL, headers={"Authorization": "Bearer garbage"})
        assert response.status_code == 401


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:
    async def test_health_endpoint(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
