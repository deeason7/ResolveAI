"""
Auth endpoint tests.

Covers: register, login, refresh, logout, /me, and key error paths.
Redis calls in refresh/logout are patched out so tests run without
a real Redis instance.
"""

import pytest
from httpx import AsyncClient
from sqlmodel import select

from app.config import settings
from app.core.security import create_access_token, create_refresh_token, hash_password
from app.database import AsyncSessionLocal
from app.models.user import User, UserRole

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
    async def test_register_creates_viewer(self, client: AsyncClient):
        registered = await client.post(REGISTER_URL, json=VALID_USER)
        profile = await client.get(
            ME_URL, headers={"Authorization": f"Bearer {registered.json()['access_token']}"}
        )
        assert profile.json()["role"] == "viewer"

    @pytest.mark.parametrize("extra", [{"role": "admin"}, {"is_active": True}])
    async def test_cannot_supply_privileged_fields(self, client: AsyncClient, extra):
        response = await client.post(REGISTER_URL, json={**VALID_USER, **extra})
        assert response.status_code == 422
        async with AsyncSessionLocal() as session:
            assert (await session.exec(select(User))).all() == []

    async def test_production_registration_defaults_disabled(self, client, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "public_registration_enabled", None)
        response = await client.post(REGISTER_URL, json=VALID_USER)
        assert response.status_code == 403
        assert "refresh_token" not in response.cookies
        assert (await client.get("/api/v1/auth/capabilities")).json() == {
            "public_registration_enabled": False
        }
        async with AsyncSessionLocal() as session:
            assert (await session.exec(select(User))).all() == []

    async def test_explicit_production_registration_still_creates_viewer(self, client, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")
        monkeypatch.setattr(settings, "public_registration_enabled", True)
        response = await client.post(REGISTER_URL, json=VALID_USER)
        assert response.status_code == 201
        async with AsyncSessionLocal() as session:
            assert (await session.exec(select(User))).one().role == UserRole.viewer

    async def test_registration_can_be_disabled_in_development(self, client, monkeypatch):
        monkeypatch.setattr(settings, "environment", "development")
        monkeypatch.setattr(settings, "public_registration_enabled", False)
        assert (await client.post(REGISTER_URL, json=VALID_USER)).status_code == 403

    async def test_demo_identity_cannot_be_self_registered(self, client):
        response = await client.post(
            REGISTER_URL, json={**VALID_USER, "email": settings.demo_email}
        )
        assert response.status_code == 403

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


class TestProvisionedRoles:
    async def test_existing_access_token_obeys_current_database_role(self, client):
        response = await client.post(REGISTER_URL, json=VALID_USER)
        token = response.json()["access_token"]
        async with AsyncSessionLocal() as session:
            user = (await session.exec(select(User).where(User.email == VALID_USER["email"]))).one()
            user.role = UserRole.analyst
            session.add(user)
            await session.commit()
            # The token deliberately claims analyst; the database decides access.
            token = create_access_token(str(user.id), "analyst")
        headers = {"Authorization": f"Bearer {token}"}
        path = "/api/v1/workspace/enqueue/classification"
        assert (await client.post(path, headers=headers)).status_code == 200
        async with AsyncSessionLocal() as session:
            user = (await session.exec(select(User).where(User.email == VALID_USER["email"]))).one()
            user.role = UserRole.viewer
            session.add(user)
            await session.commit()
        assert (await client.post(path, headers=headers)).status_code == 403

    @pytest.mark.parametrize("role", [UserRole.analyst, UserRole.admin])
    async def test_elevated_demo_denied_login_refresh_and_existing_token(self, client, role):
        async with AsyncSessionLocal() as session:
            demo = User(
                email=settings.demo_email,
                full_name="Demo",
                role=role,
                hashed_password=hash_password(VALID_USER["password"]),
            )
            session.add(demo)
            await session.commit()
        login = await client.post(
            LOGIN_URL,
            json={
                "email": settings.demo_email,
                "password": VALID_USER["password"],
            },
        )
        assert login.status_code == 403
        assert "refresh_token" not in login.cookies
        token = create_access_token(str(demo.id), role.value)
        assert (
            await client.get(ME_URL, headers={"Authorization": f"Bearer {token}"})
        ).status_code == 403
        refresh = create_refresh_token(str(demo.id))
        assert (
            await client.post(REFRESH_URL, headers={"Cookie": f"refresh_token={refresh}"})
        ).status_code == 403

    async def test_provisioned_viewer_demo_can_login(self, client):
        async with AsyncSessionLocal() as session:
            session.add(
                User(
                    email=settings.demo_email,
                    full_name="Demo",
                    role=UserRole.viewer,
                    hashed_password=hash_password(VALID_USER["password"]),
                )
            )
            await session.commit()
        response = await client.post(
            LOGIN_URL,
            json={
                "email": settings.demo_email,
                "password": VALID_USER["password"],
            },
        )
        assert response.status_code == 200


@pytest.mark.parametrize(
    "environment, enabled",
    [
        ("production", False),
        ("staging", False),
        ("typo", False),
        ("development", True),
        ("test", True),
    ],
)
def test_registration_default_is_environment_scoped(monkeypatch, environment, enabled):
    monkeypatch.setattr(settings, "environment", environment)
    monkeypatch.setattr(settings, "public_registration_enabled", None)
    assert settings.registration_enabled is enabled


def test_user_orm_default_is_viewer():
    assert (
        User(email="default@example.com", full_name="Default", hashed_password="unused").role
        == UserRole.viewer
    )


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:
    async def test_health_endpoint(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
