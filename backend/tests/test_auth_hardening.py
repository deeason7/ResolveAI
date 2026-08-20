"""
Credential-path hardening: enumeration resistance, keyed audit hashes,
blocklist key handling, and malformed-token behavior.

These assert security properties rather than business behavior, so they're
kept apart from test_api_auth.py — the same split as test_rbac_matrix.py.
"""

import hashlib
from unittest.mock import patch

import jwt
import pytest
from httpx import AsyncClient

from app.config import settings
from app.core.security import (
    create_refresh_token,
    hash_ip,
    hash_password,
    hash_token,
    token_subject,
    verify_password_or_dummy,
)

LOGIN_URL = "/api/v1/auth/login"
REGISTER_URL = "/api/v1/auth/register"
REFRESH_URL = "/api/v1/auth/refresh"

USER = {"email": "hardening@example.com", "full_name": "H", "password": "securepassword123"}


class TestNoUserEnumeration:
    """A missing account must be indistinguishable from a wrong password."""

    def test_absent_hash_still_runs_bcrypt(self):
        """The None branch has to do the work, or timing leaks the account list."""
        with patch("app.core.security.bcrypt.checkpw", return_value=False) as checkpw:
            assert verify_password_or_dummy("anything", None) is False
        assert checkpw.call_count == 1, "no bcrypt call means the miss returns early"

    def test_matching_password_verifies(self):
        h = hash_password("securepassword123")
        assert verify_password_or_dummy("securepassword123", h) is True

    def test_wrong_password_rejected(self):
        h = hash_password("securepassword123")
        assert verify_password_or_dummy("wrong", h) is False

    async def test_unknown_email_and_bad_password_are_identical(self, client: AsyncClient):
        await client.post(REGISTER_URL, json=USER)

        unknown = await client.post(
            LOGIN_URL, json={"email": "nobody@example.com", "password": "securepassword123"}
        )
        wrong = await client.post(
            LOGIN_URL, json={"email": USER["email"], "password": "not-the-password"}
        )

        assert unknown.status_code == wrong.status_code == 401
        assert unknown.json() == wrong.json()


class TestAuditIpHashing:
    def test_is_keyed_not_a_bare_digest(self):
        """A plain sha256 of an IP is reversible — 2^32 is a small table."""
        ip = "203.0.113.7"
        assert hash_ip(ip) != hashlib.sha256(ip.encode()).hexdigest()

    def test_is_deterministic(self):
        assert hash_ip("203.0.113.7") == hash_ip("203.0.113.7")

    def test_distinct_ips_differ(self):
        assert hash_ip("203.0.113.7") != hash_ip("203.0.113.8")

    def test_key_changes_the_digest(self):
        before = hash_ip("203.0.113.7")
        with patch.object(settings, "jwt_secret_key", settings.jwt_secret_key + "-rotated"):
            assert hash_ip("203.0.113.7") != before


class TestTokenSubject:
    def test_parses_a_uuid(self):
        import uuid

        u = uuid.uuid4()
        assert token_subject({"sub": str(u)}) == u

    @pytest.mark.parametrize("sub", ["not-a-uuid", "", None, 12345, {"nested": 1}])
    def test_malformed_subject_returns_none(self, sub):
        assert token_subject({"sub": sub}) is None

    def test_missing_subject_returns_none(self):
        assert token_subject({}) is None

    async def test_signed_token_with_junk_subject_is_401_not_500(self, client: AsyncClient):
        """Signed but malformed: bare uuid.UUID() here would surface as a 500."""
        token = jwt.encode(
            {"sub": "definitely-not-a-uuid", "role": "analyst", "type": "access"},
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        resp = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401


class TestBlocklistKeys:
    def test_token_is_hashed_not_stored_raw(self):
        token = create_refresh_token("00000000-0000-0000-0000-000000000001")
        assert token not in hash_token(token)
        assert len(hash_token(token)) == 64

    async def test_legacy_raw_key_still_revokes(self, client: AsyncClient, fake_redis):
        """Entries written before hashed keys stay honored until they expire."""
        from app.api.routes.auth import REFRESH_BLOCKLIST_PREFIX

        reg = await client.post(REGISTER_URL, json={**USER, "email": "legacy@example.com"})
        token = reg.cookies["refresh_token"]

        await fake_redis.setex(f"{REFRESH_BLOCKLIST_PREFIX}{token}", 60, "revoked")

        # The cookie is issued Secure (ENVIRONMENT != development), so the test
        # transport won't replay it over http — hand it over explicitly.
        resp = await client.post(REFRESH_URL, headers={"Cookie": f"refresh_token={token}"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Token revoked"

    async def test_hashed_key_revokes(self, client: AsyncClient, fake_redis):
        """The key the code writes today must also be honored on read."""
        from app.api.routes.auth import REFRESH_BLOCKLIST_PREFIX

        reg = await client.post(REGISTER_URL, json={**USER, "email": "hashed@example.com"})
        token = reg.cookies["refresh_token"]

        await fake_redis.setex(f"{REFRESH_BLOCKLIST_PREFIX}{hash_token(token)}", 60, "revoked")

        resp = await client.post(REFRESH_URL, headers={"Cookie": f"refresh_token={token}"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Token revoked"

    async def test_valid_token_still_refreshes(self, client: AsyncClient):
        """Guard against the blocklist rejecting everything."""
        reg = await client.post(REGISTER_URL, json={**USER, "email": "fresh@example.com"})
        token = reg.cookies["refresh_token"]

        resp = await client.post(REFRESH_URL, headers={"Cookie": f"refresh_token={token}"})
        assert resp.status_code == 200
        assert "access_token" in resp.json()


class TestCredentialRateLimit:
    async def test_login_is_capped_below_the_global_default(self, client: AsyncClient):
        """Every attempt costs a bcrypt verify, so the endpoint needs its own ceiling."""
        from app.api.routes.auth import _CREDENTIAL_RATE_LIMIT

        allowed = int(_CREDENTIAL_RATE_LIMIT.split("/")[0])
        body = {"email": "nobody@example.com", "password": "securepassword123"}

        for _ in range(allowed):
            assert (await client.post(LOGIN_URL, json=body)).status_code == 401

        assert (await client.post(LOGIN_URL, json=body)).status_code == 429
