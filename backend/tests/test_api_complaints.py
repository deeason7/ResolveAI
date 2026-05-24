"""
Complaint endpoint tests — submit / list / get / bulk-import.

Bulk-import tests monkey-patch ALLOWED_IMPORT_ROOTS to include the
pytest tmp_path so we don't have to write into the container's
/fine_tuning mount from the test process.
"""

import csv
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

REGISTER_URL = "/api/v1/auth/register"
LOGIN_URL = "/api/v1/auth/login"
COMPLAINTS_URL = "/api/v1/complaints/"
BULK_IMPORT_URL = "/api/v1/complaints/bulk-import"

VALID_USER = {
    "email": "analyst@test.com",
    "full_name": "Test Analyst",
    "password": "securepassword123",
}
ADMIN_USER = {
    "email": "admin@test.com",
    "full_name": "Test Admin",
    "password": "securepassword123",
}

VALID_COMPLAINT = {
    "narrative": "I have been charged a fee that was not disclosed when I opened the account.",
    "product": "Checking account",
    "issue": "Fees",
    "company": "BigBank",
    "state": "CA",
}


# ── helpers ───────────────────────────────────────────────────────────────────


async def _register_and_token(client: AsyncClient, user: dict) -> str:
    """Register a fresh user and return their access token."""
    r = await client.post(REGISTER_URL, json=user)
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


async def _promote_to_admin(email: str) -> None:
    """Direct DB write to flip a user's role to admin."""
    from app.database import engine
    from app.models.user import User, UserRole

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        result = await s.exec(select(User).where(User.email == email))
        user = result.first()
        assert user is not None
        user.role = UserRole.admin
        s.add(user)
        await s.commit()


@pytest_asyncio.fixture()
async def analyst_token(client: AsyncClient) -> str:
    return await _register_and_token(client, VALID_USER)


@pytest_asyncio.fixture()
async def admin_token(client: AsyncClient) -> str:
    token = await _register_and_token(client, ADMIN_USER)
    await _promote_to_admin(ADMIN_USER["email"])
    return token


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── submit ────────────────────────────────────────────────────────────────────


class TestSubmit:
    async def test_submit_returns_201_with_id(self, client: AsyncClient, analyst_token: str):
        r = await client.post(COMPLAINTS_URL, json=VALID_COMPLAINT, headers=_auth(analyst_token))
        assert r.status_code == 201, r.text
        body = r.json()
        assert "id" in body
        assert body["narrative"] == VALID_COMPLAINT["narrative"]
        assert body["state"] == "CA"
        assert body["status"] == "pending"

    async def test_submit_requires_auth(self, client: AsyncClient):
        r = await client.post(COMPLAINTS_URL, json=VALID_COMPLAINT)
        assert r.status_code == 401

    async def test_submit_rejects_short_narrative(
        self, client: AsyncClient, analyst_token: str
    ):
        body = {**VALID_COMPLAINT, "narrative": "too short"}
        r = await client.post(COMPLAINTS_URL, json=body, headers=_auth(analyst_token))
        assert r.status_code == 422

    async def test_submit_uppercases_state(self, client: AsyncClient, analyst_token: str):
        body = {**VALID_COMPLAINT, "state": "ny"}
        r = await client.post(COMPLAINTS_URL, json=body, headers=_auth(analyst_token))
        # state has min=2 max=2; we send lowercase, route uppercases on insert
        assert r.status_code == 201
        assert r.json()["state"] == "NY"


# ── get by id ─────────────────────────────────────────────────────────────────


class TestGetById:
    async def test_get_returns_complaint(self, client: AsyncClient, analyst_token: str):
        created = await client.post(
            COMPLAINTS_URL, json=VALID_COMPLAINT, headers=_auth(analyst_token)
        )
        cid = created.json()["id"]

        r = await client.get(f"{COMPLAINTS_URL}{cid}", headers=_auth(analyst_token))
        assert r.status_code == 200
        assert r.json()["id"] == cid

    async def test_unknown_id_returns_404(self, client: AsyncClient, analyst_token: str):
        r = await client.get(
            f"{COMPLAINTS_URL}00000000-0000-0000-0000-000000000000",
            headers=_auth(analyst_token),
        )
        assert r.status_code == 404

    async def test_bad_uuid_returns_422(self, client: AsyncClient, analyst_token: str):
        r = await client.get(f"{COMPLAINTS_URL}not-a-uuid", headers=_auth(analyst_token))
        assert r.status_code == 422


# ── list / pagination / filters ───────────────────────────────────────────────


class TestList:
    async def test_empty_list(self, client: AsyncClient, analyst_token: str):
        r = await client.get(COMPLAINTS_URL, headers=_auth(analyst_token))
        assert r.status_code == 200
        body = r.json()
        assert body == {"items": [], "total": 0, "limit": 50, "offset": 0}

    async def test_list_filters_and_paginates(self, client: AsyncClient, analyst_token: str):
        # Seed 3 complaints across two products
        for prod in ["Mortgage", "Mortgage", "Credit card"]:
            await client.post(
                COMPLAINTS_URL,
                json={**VALID_COMPLAINT, "product": prod},
                headers=_auth(analyst_token),
            )

        # No filter → 3
        r = await client.get(COMPLAINTS_URL, headers=_auth(analyst_token))
        assert r.json()["total"] == 3

        # Filter by product → 2
        r = await client.get(
            COMPLAINTS_URL, params={"product": "Mortgage"}, headers=_auth(analyst_token)
        )
        body = r.json()
        assert body["total"] == 2
        assert len(body["items"]) == 2
        assert all(item["product"] == "Mortgage" for item in body["items"])

        # Pagination — limit=1 returns 1 item but total still 3
        r = await client.get(
            COMPLAINTS_URL, params={"limit": 1}, headers=_auth(analyst_token)
        )
        body = r.json()
        assert body["total"] == 3
        assert len(body["items"]) == 1
        assert body["limit"] == 1


# ── bulk import ───────────────────────────────────────────────────────────────


class TestBulkImport:
    async def test_requires_admin(self, client: AsyncClient, analyst_token: str):
        r = await client.post(
            BULK_IMPORT_URL,
            json={"path": "/fine_tuning/data/raw/anything.csv"},
            headers=_auth(analyst_token),
        )
        assert r.status_code == 403

    async def test_requires_auth(self, client: AsyncClient):
        r = await client.post(
            BULK_IMPORT_URL, json={"path": "/fine_tuning/data/raw/anything.csv"}
        )
        assert r.status_code == 401

    async def test_rejects_path_outside_sandbox(
        self, client: AsyncClient, admin_token: str
    ):
        r = await client.post(
            BULK_IMPORT_URL,
            json={"path": "/etc/passwd"},
            headers=_auth(admin_token),
        )
        assert r.status_code == 400

    async def test_missing_file_returns_404(
        self, client: AsyncClient, admin_token: str, tmp_path: Path, monkeypatch
    ):
        # Allow tmp_path so the sandbox check passes, then point at a non-existent file
        from app.api.routes import complaints as cr

        monkeypatch.setattr(cr, "ALLOWED_IMPORT_ROOTS", (tmp_path,))
        r = await client.post(
            BULK_IMPORT_URL,
            json={"path": str(tmp_path / "missing.csv")},
            headers=_auth(admin_token),
        )
        assert r.status_code == 404

    async def test_successful_import(
        self, client: AsyncClient, admin_token: str, tmp_path: Path, monkeypatch
    ):
        from app.api.routes import complaints as cr

        monkeypatch.setattr(cr, "ALLOWED_IMPORT_ROOTS", (tmp_path,))

        csv_path = tmp_path / "tiny.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "complaint_id",
                    "consumer_complaint_narrative",
                    "product",
                    "sub_product",
                    "issue",
                    "sub_issue",
                    "company",
                    "company_response",
                    "state",
                    "date_received",
                ],
            )
            w.writeheader()
            w.writerow({
                "complaint_id": "1001",
                "consumer_complaint_narrative": "I was charged a late fee even though I paid on time.",
                "product": "Credit card",
                "sub_product": "", "issue": "Fee", "sub_issue": "",
                "company": "BankCo", "company_response": "In progress",
                "state": "CA", "date_received": "2024-01-15",
            })
            w.writerow({
                "complaint_id": "1002",
                "consumer_complaint_narrative": "My mortgage payment was misapplied to the wrong account.",
                "product": "Mortgage",
                "sub_product": "", "issue": "Payment", "sub_issue": "",
                "company": "MortgageCo", "company_response": "Closed",
                "state": "NY", "date_received": "2024-02-20",
            })
            w.writerow({
                "complaint_id": "",   # missing source id → still ingestible (NULL allowed)
                "consumer_complaint_narrative": "Funds transfer never reached the recipient.",
                "product": "Money transfer",
                "sub_product": "", "issue": "Delay", "sub_issue": "",
                "company": "TransferCo", "company_response": "In progress",
                "state": "TX", "date_received": "2024-03-10",
            })

        r = await client.post(
            BULK_IMPORT_URL,
            json={"path": str(csv_path), "batch_size": 100},
            headers=_auth(admin_token),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["rows_read"] == 3
        assert body["rows_inserted"] == 3
        assert body["rows_skipped"] == 0
        assert body["batches"] == 1

        # Verify they're actually in the listing
        listed = await client.get(COMPLAINTS_URL, headers=_auth(admin_token))
        assert listed.json()["total"] == 3

    async def test_import_is_idempotent(
        self, client: AsyncClient, admin_token: str, tmp_path: Path, monkeypatch
    ):
        from app.api.routes import complaints as cr

        monkeypatch.setattr(cr, "ALLOWED_IMPORT_ROOTS", (tmp_path,))

        csv_path = tmp_path / "dup.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "complaint_id", "consumer_complaint_narrative", "product",
                    "sub_product", "issue", "sub_issue", "company",
                    "company_response", "state", "date_received",
                ],
            )
            w.writeheader()
            w.writerow({
                "complaint_id": "9001",
                "consumer_complaint_narrative": "A complaint that we will try to import twice.",
                "product": "Checking account",
                "sub_product": "", "issue": "Fees", "sub_issue": "",
                "company": "BankCo", "company_response": "Closed",
                "state": "CA", "date_received": "2024-04-01",
            })

        first = await client.post(
            BULK_IMPORT_URL,
            json={"path": str(csv_path), "batch_size": 100},
            headers=_auth(admin_token),
        )
        assert first.json()["rows_inserted"] == 1

        second = await client.post(
            BULK_IMPORT_URL,
            json={"path": str(csv_path), "batch_size": 100},
            headers=_auth(admin_token),
        )
        assert second.status_code == 200
        assert second.json()["rows_read"] == 1
        assert second.json()["rows_inserted"] == 0   # ON CONFLICT skipped it
