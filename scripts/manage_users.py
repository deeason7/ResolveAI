"""Inspect or explicitly provision application accounts; dry-run unless --apply.

This command never loads .env or app.database. Select the target through an
explicit environment-variable name. Passwords come from a hidden prompt or a
named environment variable, never a command argument or the JSON report.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import bcrypt
from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.models.audit_log import AuditLog  # noqa: E402
from app.models.user import User, UserRole  # noqa: E402
from app.schemas.auth import RegisterRequest  # noqa: E402


def _email(value: str) -> str:
    try:
        return str(TypeAdapter(EmailStr).validate_python(value))
    except ValidationError:
        raise ValueError("A valid email selector is required") from None


def _public(user: User) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "role": user.role.value,
        "is_active": user.is_active,
    }


def _check_demo_role(email: str, role: UserRole, demo_email: str) -> None:
    if email.casefold() == demo_email.casefold() and role != UserRole.viewer:
        raise ValueError("The configured demo identity may only have the viewer role")


async def inventory(
    session: AsyncSession,
    *,
    email: str | None = None,
    user_id: uuid.UUID | None = None,
    role: UserRole | None = None,
    limit: int = 100,
) -> dict:
    if sum(value is not None for value in (email, user_id, role)) != 1:
        raise ValueError("Choose exactly one email, user ID, or role selector")
    if not 1 <= limit <= 200:
        raise ValueError("Inventory limit must be between 1 and 200")
    statement = select(User).order_by(User.id)
    if email is not None:
        statement = statement.where(User.email == _email(email))
    elif user_id is not None:
        statement = statement.where(User.id == user_id)
    else:
        statement = statement.where(User.role == role)
    rows = (await session.exec(statement.limit(limit + 1))).all()
    return {"users": [_public(row) for row in rows[:limit]], "truncated": len(rows) > limit}


async def create_user(
    session: AsyncSession,
    *,
    email: str,
    full_name: str,
    role: UserRole,
    demo_email: str,
    apply: bool = False,
    password_reader: Callable[[], str] | None = None,
) -> dict:
    email = _email(email)
    if not full_name.strip() or len(full_name) > 255:
        raise ValueError("Full name must contain 1 to 255 characters")
    _check_demo_role(email, role, demo_email)
    existing = (await session.exec(select(User).where(User.email == email))).first()
    if existing is not None:
        if existing.role != role:
            raise ValueError("Existing role differs; inspect its ID and use set-role explicitly")
        return {"action": "unchanged", "applied": False, "user": _public(existing)}
    if not apply:
        return {"action": "would_create", "applied": False, "email": email, "role": role.value}
    if password_reader is None:
        raise ValueError("A password source is required for a new account")
    password = password_reader()
    if len(password.encode("utf-8")) > 72:
        raise ValueError("New bcrypt passwords must not exceed 72 UTF-8 bytes")
    try:
        RegisterRequest(email=email, full_name=full_name, password=password)
    except ValidationError:
        raise ValueError("New account fields do not meet registration requirements") from None
    hashed = await asyncio.to_thread(bcrypt.hashpw, password.encode(), bcrypt.gensalt())
    user = User(
        email=email,
        full_name=full_name,
        role=role,
        hashed_password=hashed.decode(),
    )
    session.add(user)
    await session.flush()
    session.add(
        AuditLog(
            event="account_provisioned",
            metadata_={
                "target_user_id": str(user.id),
                "role": role.value,
                "source": "manage_users",
            },
        )
    )
    return {"action": "created", "applied": True, "user": _public(user)}


async def set_role(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    expected_role: UserRole,
    role: UserRole,
    demo_email: str,
    apply: bool = False,
) -> dict:
    user = (await session.exec(select(User).where(User.id == user_id).with_for_update())).first()
    if user is None:
        raise ValueError("Selected user does not exist")
    _check_demo_role(user.email, role, demo_email)
    if user.role == role:
        return {"action": "unchanged", "applied": False, "user": _public(user)}
    if user.role != expected_role:
        raise ValueError("Current role differs from expected role; inspect the user again")
    before = _public(user)
    if apply:
        user.role = role
        user.updated_at = datetime.utcnow()
        session.add(user)
        session.add(
            AuditLog(
                event="account_role_changed",
                metadata_={
                    "target_user_id": str(user.id),
                    "previous_role": expected_role.value,
                    "role": role.value,
                    "source": "manage_users",
                },
            )
        )
    return {
        "action": "role_changed" if apply else "would_change_role",
        "applied": apply,
        "before": before,
        "requested_role": role.value,
    }


async def reset_password(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    expected_email: str,
    reason: str,
    apply: bool = False,
    password_reader: Callable[[], str] | None = None,
) -> dict:
    """Recover one confirmed account; never guess which old hashes were truncated."""
    if not reason.strip() or len(reason) > 500:
        raise ValueError("A reset reason between 1 and 500 characters is required")
    user = (await session.exec(select(User).where(User.id == user_id).with_for_update())).first()
    if user is None or user.email != _email(expected_email):
        raise ValueError("Selected user ID and expected email do not match an account")
    report = {
        "action": "password_reset" if apply else "would_reset_password",
        "applied": apply,
        "user": _public(user),
        "session_effect": (
            "Existing refresh sessions are invalidated on commit; issued access tokens "
            "remain valid until expiry unless the account is disabled"
        ),
    }
    if not apply:
        return report
    if password_reader is None:
        raise ValueError("A password source is required for a reset")
    password = password_reader()
    try:
        RegisterRequest(email=user.email, full_name=user.full_name, password=password)
    except ValidationError:
        raise ValueError(
            "New password must contain at least 8 characters and at most 72 UTF-8 bytes"
        ) from None
    hashed = await asyncio.to_thread(bcrypt.hashpw, password.encode(), bcrypt.gensalt())
    user.hashed_password = hashed.decode()
    user.updated_at = datetime.utcnow()
    session.add(user)
    session.add(
        AuditLog(
            event="account_password_reset",
            metadata_={
                "target_user_id": str(user.id),
                "reason": reason.strip(),
                "source": "manage_users",
            },
        )
    )
    return report


def _password_reader(env_name: str | None) -> str:
    if env_name:
        value = os.environ.get(env_name)
        if not value:
            raise ValueError("The selected password environment variable is empty")
        return value
    if not sys.stdin.isatty():
        raise ValueError("Use a terminal or --password-env for account password changes")
    password = getpass.getpass("New password: ")
    if password != getpass.getpass("Repeat password: "):
        raise ValueError("Passwords do not match")
    return password


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--database-url-env",
        required=True,
        metavar="NAME",
        help="Name of the environment variable containing the target URL",
    )
    command.add_argument("--require-ssl", action="store_true", help="Require PostgreSQL TLS")
    command.add_argument(
        "--demo-email", default=os.environ.get("DEMO_EMAIL", "demo@resolveai-demo.com")
    )
    actions = command.add_subparsers(dest="action", required=True)
    inspect = actions.add_parser("inventory", help="Read a bounded account inventory")
    selectors = inspect.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--email")
    selectors.add_argument("--user-id", type=uuid.UUID)
    selectors.add_argument("--role", type=UserRole, choices=list(UserRole))
    inspect.add_argument("--limit", type=int, default=100)
    create = actions.add_parser("create", help="Create an account; existing accounts are unchanged")
    create.add_argument("--email", required=True)
    create.add_argument("--full-name", required=True)
    create.add_argument("--role", type=UserRole, choices=list(UserRole), required=True)
    create.add_argument("--password-env", metavar="NAME")
    create.add_argument("--apply", action="store_true")
    change = actions.add_parser("set-role", help="Change one explicitly selected account role")
    change.add_argument("--user-id", type=uuid.UUID, required=True)
    change.add_argument("--expected-role", type=UserRole, choices=list(UserRole), required=True)
    change.add_argument("--role", type=UserRole, choices=list(UserRole), required=True)
    change.add_argument("--apply", action="store_true")
    reset = actions.add_parser("reset-password", help="Reset one confirmed account password")
    reset.add_argument("--user-id", type=uuid.UUID, required=True)
    reset.add_argument("--expected-email", required=True)
    reset.add_argument("--reason", required=True)
    reset.add_argument("--password-env", metavar="NAME")
    reset.add_argument("--apply", action="store_true")
    return command


async def run(args: argparse.Namespace) -> dict:
    url = os.environ.get(args.database_url_env)
    if not url:
        raise ValueError("The explicitly selected database URL environment variable is empty")
    kwargs = {"connect_args": {"ssl": True}} if args.require_ssl else {}
    engine = create_async_engine(url, echo=False, **kwargs)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with factory() as session, session.begin():
            if args.action == "inventory":
                return await inventory(
                    session,
                    email=args.email,
                    user_id=args.user_id,
                    role=args.role,
                    limit=args.limit,
                )
            if args.action == "create":
                return await create_user(
                    session,
                    email=args.email,
                    full_name=args.full_name,
                    role=args.role,
                    demo_email=args.demo_email,
                    apply=args.apply,
                    password_reader=lambda: _password_reader(args.password_env),
                )
            if args.action == "reset-password":
                return await reset_password(
                    session,
                    user_id=args.user_id,
                    expected_email=args.expected_email,
                    reason=args.reason,
                    apply=args.apply,
                    password_reader=lambda: _password_reader(args.password_env),
                )
            return await set_role(
                session,
                user_id=args.user_id,
                expected_role=args.expected_role,
                role=args.role,
                demo_email=args.demo_email,
                apply=args.apply,
            )
    finally:
        await engine.dispose()


def main() -> int:
    args = parser().parse_args()
    try:
        result = asyncio.run(run(args))
    except ValueError as exc:
        print(f"Account operation refused: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Driver/SQL exception text can contain credentials or bound password hashes.
        print(
            "Account operation failed; verify the selected database, schema and connectivity",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
