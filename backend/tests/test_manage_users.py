"""Account operations use synthetic rows; no operator database or password is loaded."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from sqlmodel import select

from app.database import AsyncSessionLocal
from app.models.audit_log import AuditLog
from app.models.user import User, UserRole

spec = importlib.util.spec_from_file_location(
    "manage_users", Path(__file__).resolve().parents[2] / "scripts" / "manage_users.py"
)
manage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manage)

DEMO = "demo@example.com"
PASSWORD = "synthetic-password-only"


async def _seed(role=UserRole.analyst, email="staff@example.com"):
    async with AsyncSessionLocal() as session, session.begin():
        user = User(email=email, full_name="Staff", role=role, hashed_password="synthetic-hash")
        session.add(user)
        await session.flush()
        return user.id


async def test_create_is_dry_run_without_reading_password(client):
    def forbidden_password_read():
        pytest.fail("Dry run must not read a password")

    async with AsyncSessionLocal() as session, session.begin():
        result = await manage.create_user(
            session,
            email=DEMO,
            full_name="Demo",
            role=UserRole.viewer,
            demo_email=DEMO,
            password_reader=forbidden_password_read,
        )
        assert result["action"] == "would_create"
        assert (await session.exec(select(User))).all() == []
        assert (await session.exec(select(AuditLog))).all() == []


async def test_create_is_idempotent_and_preserves_existing_account(client):
    async with AsyncSessionLocal() as session, session.begin():
        created = await manage.create_user(
            session,
            email=DEMO,
            full_name="Demo",
            role=UserRole.viewer,
            demo_email=DEMO,
            apply=True,
            password_reader=lambda: PASSWORD,
        )
    async with AsyncSessionLocal() as session, session.begin():
        again = await manage.create_user(
            session,
            email=DEMO,
            full_name="Other name",
            role=UserRole.viewer,
            demo_email=DEMO,
            apply=True,
            password_reader=lambda: pytest.fail("Do not replace existing credentials"),
        )
        rows = (await session.exec(select(User))).all()
        assert len(rows) == 1
        assert rows[0].full_name == "Demo"
        assert len((await session.exec(select(AuditLog))).all()) == 1
        assert created["user"]["id"] == again["user"]["id"]
        assert again["action"] == "unchanged"
        report = json.dumps([created, again])
        assert PASSWORD not in report
        assert rows[0].hashed_password not in report


async def test_existing_role_requires_explicit_change(client):
    await _seed()
    async with AsyncSessionLocal() as session:
        with pytest.raises(ValueError, match="set-role explicitly"):
            await manage.create_user(
                session,
                email="staff@example.com",
                full_name="Staff",
                role=UserRole.viewer,
                demo_email=DEMO,
                apply=True,
            )


async def test_role_change_is_selected_guarded_and_idempotent(client):
    selected = await _seed()
    unrelated = await _seed(email="other@example.com")
    async with AsyncSessionLocal() as session, session.begin():
        report = await manage.set_role(
            session,
            user_id=selected,
            expected_role=UserRole.analyst,
            role=UserRole.viewer,
            demo_email=DEMO,
        )
        assert report["action"] == "would_change_role"
        assert (await session.get(User, selected)).role == UserRole.analyst
        with pytest.raises(ValueError, match="expected role"):
            await manage.set_role(
                session,
                user_id=selected,
                expected_role=UserRole.admin,
                role=UserRole.viewer,
                demo_email=DEMO,
                apply=True,
            )
        await manage.set_role(
            session,
            user_id=selected,
            expected_role=UserRole.analyst,
            role=UserRole.viewer,
            demo_email=DEMO,
            apply=True,
        )
    async with AsyncSessionLocal() as session, session.begin():
        repeated = await manage.set_role(
            session,
            user_id=selected,
            expected_role=UserRole.analyst,
            role=UserRole.viewer,
            demo_email=DEMO,
            apply=True,
        )
        assert repeated["action"] == "unchanged"
        assert (await session.get(User, unrelated)).role == UserRole.analyst
        assert len((await session.exec(select(AuditLog))).all()) == 1


async def test_demo_cannot_be_provisioned_or_promoted_as_writer(client):
    selected = await _seed(role=UserRole.viewer, email=DEMO)
    async with AsyncSessionLocal() as session:
        with pytest.raises(ValueError, match="viewer role"):
            await manage.create_user(
                session,
                email=DEMO,
                full_name="Demo",
                role=UserRole.admin,
                demo_email=DEMO,
                apply=True,
            )
        with pytest.raises(ValueError, match="viewer role"):
            await manage.set_role(
                session,
                user_id=selected,
                expected_role=UserRole.viewer,
                role=UserRole.analyst,
                demo_email=DEMO,
                apply=True,
            )


async def test_inventory_is_bounded_and_never_returns_credentials(client):
    await _seed()
    await _seed(email="other@example.com")
    async with AsyncSessionLocal() as session:
        report = await manage.inventory(session, role=UserRole.analyst, limit=1)
        assert report["truncated"] is True
        assert len(report["users"]) == 1
        assert set(report["users"][0]) == {"id", "email", "role", "is_active"}
        with pytest.raises(ValueError, match="exactly one"):
            await manage.inventory(session)
        with pytest.raises(ValueError, match="between"):
            await manage.inventory(session, role=UserRole.analyst, limit=10000)


async def test_new_password_boundaries_fail_without_echoing_secret(client):
    password = "é" * 40
    async with AsyncSessionLocal() as session:
        with pytest.raises(ValueError, match="72 UTF-8 bytes") as error:
            await manage.create_user(
                session,
                email=DEMO,
                full_name="Demo",
                role=UserRole.viewer,
                demo_email=DEMO,
                apply=True,
                password_reader=lambda: password,
            )
        assert password not in str(error.value)
        assert (await session.exec(select(User))).all() == []


def test_cli_requires_explicit_database_selection(monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "not-used")
    monkeypatch.delenv("MISSING_ADMIN_TARGET", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage_users.py",
            "--database-url-env",
            "MISSING_ADMIN_TARGET",
            "inventory",
            "--role",
            "viewer",
        ],
    )
    assert manage.main() == 1
    captured = capsys.readouterr()
    assert "not-used" not in captured.err
    assert "explicitly selected" in captured.err


def test_noninteractive_creation_requires_password_env(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError, match="terminal"):
        manage._password_reader(None)
