"""Migrations, relational constraints, and controlled independent transactions."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import MetaData, func, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.audit_log import AuditLog
from app.models.complaint import Complaint
from app.models.complaint_label import ComplaintLabel
from app.models.resolution import Resolution
from app.models.user import User

pytestmark = pytest.mark.postgres
BACKEND = Path(__file__).resolve().parents[2]


async def test_empty_database_migrates_to_head(pg_engine):
    script = ScriptDirectory.from_config(Config(str(BACKEND / "alembic.ini")))
    async with pg_engine.connect() as connection:
        tables = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
    assert set(tables) == {
        "alembic_version",
        "audit_logs",
        "complaint_labels",
        "complaints",
        "llm_logs",
        "resolutions",
        "review_events",
        "users",
    }
    assert revision == script.get_current_head()


async def test_raw_inserts_use_safe_defaults_without_overriding_explicit_roles(pg_engine):
    async with pg_engine.begin() as connection:
        for role in (None, "admin", "analyst", "viewer"):
            values = {
                "id": uuid.uuid4(),
                "email": f"{role}@example.test",
                "role": role,
            }
            # Raw SQL bypasses all Python/Pydantic defaults.
            role_column = ", role" if role is not None else ""
            role_value = ", :role" if role is not None else ""
            actual = await connection.scalar(
                text(
                    "INSERT INTO users (id, email, full_name, hashed_password, created_at, "
                    f"updated_at{role_column}) VALUES (:id, :email, 'Fixture', 'unused', "
                    f"CURRENT_TIMESTAMP, CURRENT_TIMESTAMP{role_value}) RETURNING role"
                ),
                values,
            )
            assert actual == (role or "viewer")
        row = (
            await connection.execute(
                text(
                    "INSERT INTO complaints (id, narrative, created_at, updated_at) "
                    "VALUES (:id, 'New synthetic intake', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                    "RETURNING classification_outcome, classification_stale, classification_source, "
                    "classification_error, classification_latency_ms, sentiment, urgency, priority_score"
                ),
                {"id": uuid.uuid4()},
            )
        ).one()
        assert tuple(row) == ("pending", False, None, None, None, None, None, None)


async def test_previous_revision_upgrades_without_losing_linked_rows(postgres_database):
    await postgres_database.upgrade("d4f8c2b7e1a9")
    sessions = async_sessionmaker(
        postgres_database.engine, class_=AsyncSession, expire_on_commit=False
    )
    # Reflect the historical schema: the current ORM may require columns that
    # did not exist in this revision. This fixture must survive future changes.
    from datetime import datetime

    complaint_id, user_id, label_id, draft_id, audit_id = [uuid.uuid4() for _ in range(5)]
    now = datetime(2026, 1, 2)
    stamps = {"created_at": now, "updated_at": now}
    async with postgres_database.engine.begin() as conn:
        metadata = MetaData()
        await conn.run_sync(metadata.reflect)
        tables = metadata.tables
        await conn.execute(
            tables["complaints"]
            .insert()
            .values(
                id=complaint_id,
                narrative="Synthetic reporting dispute",
                cfpb_complaint_id="migration-1",
                sentiment="negative",
                urgency=4,
                status="resolved",
                **stamps,
            )
        )
        await conn.execute(
            tables["users"]
            .insert()
            .values(
                id=user_id,
                email="migration@example.test",
                full_name="Test reviewer",
                hashed_password="unused",
                **stamps,
            )
        )
        await conn.execute(
            tables["complaint_labels"]
            .insert()
            .values(
                id=label_id,
                complaint_id=complaint_id,
                label_source="synthetic-test",
                sentiment="negative",
                intent="dispute_resolution",
                urgency=4,
                reasoning="Fixture, not model output",
                key_entities=[],
                labeled_at=now,
                **stamps,
            )
        )
        await conn.execute(
            tables["resolutions"]
            .insert()
            .values(
                id=draft_id,
                complaint_id=complaint_id,
                draft_text="Synthetic draft retained through upgrade",
                guardrail_violations=[{"code": "fixture"}],
                **stamps,
            )
        )
        await conn.execute(
            tables["audit_logs"]
            .insert()
            .values(
                id=audit_id,
                user_id=user_id,
                event="migration_fixture",
                created_at=now,
            )
        )

    await postgres_database.upgrade()

    async with sessions() as session:
        restored = await session.get(Resolution, draft_id)
        assert restored.complaint_id == complaint_id
        assert restored.draft_text == "Synthetic draft retained through upgrade"
        assert restored.guardrail_violations == [{"code": "fixture"}]
        assert (await session.get(ComplaintLabel, label_id)).complaint_id == complaint_id
        assert (await session.get(AuditLog, audit_id)).user_id == user_id
        legacy = await session.get(Complaint, complaint_id)
        assert legacy.legacy_approval_outcome_unknown is True
        assert legacy.case_version == legacy.input_revision == 1
        assert restored.legacy_provenance is True
        assert restored.input_revision == 1
        assert legacy.classification_outcome == "legacy_unknown"
        assert legacy.classification_source is None
        assert legacy.classified_at is None
        assert legacy.classification_attempted_at is None
        assert (legacy.sentiment, legacy.urgency, legacy.status) == ("negative", 4, "resolved")
        # Changing the default must never promote/demote an existing account.
        assert (await session.get(User, user_id)).role == "analyst"
        session.add(Resolution(complaint_id=complaint_id, version=1, draft_text="Duplicate"))
        with pytest.raises(IntegrityError) as caught:
            await session.commit()
        assert caught.value.orig.sqlstate == "23505"
        await session.rollback()


@pytest.mark.parametrize("child", ["resolution", "audit"])
async def test_foreign_keys_reject_missing_parents(pg_sessionmaker, child):
    row = (
        Resolution(complaint_id=uuid.uuid4(), draft_text="Orphan")
        if child == "resolution"
        else AuditLog(user_id=uuid.uuid4(), event="orphan")
    )
    async with pg_sessionmaker() as session:
        session.add(row)
        with pytest.raises(IntegrityError) as caught:
            await session.commit()
        assert caught.value.orig.sqlstate == "23503"
        await session.rollback()


async def test_source_identity_unique_constraint(pg_sessionmaker):
    async with pg_sessionmaker() as session:
        session.add(Complaint(narrative="Original", cfpb_complaint_id="unique-test"))
        await session.commit()
        session.add(Complaint(narrative="Duplicate", cfpb_complaint_id="unique-test"))
        with pytest.raises(IntegrityError) as caught:
            await session.commit()
        assert caught.value.orig.sqlstate == "23505"
        await session.rollback()


async def test_concurrent_draft_version_race_uses_independent_connections(pg_sessionmaker):
    """Both writers read the same next version; the database admits only one.

    This proves the constraint, not that the current API handles the loser well.
    The barrier fixes the race ordering without relying on scheduling delays.
    """
    complaint = Complaint(narrative="Concurrent synthetic case")
    async with pg_sessionmaker() as session:
        session.add(complaint)
        await session.flush()
        session.add(Resolution(complaint_id=complaint.id, draft_text="Version 1"))
        await session.commit()

    ready = asyncio.Barrier(2)

    async def writer():
        async with pg_sessionmaker() as session:
            pid = await session.scalar(text("SELECT pg_backend_pid()"))
            previous = await session.scalar(
                select(func.max(Resolution.version)).where(Resolution.complaint_id == complaint.id)
            )
            version = previous + 1
            await ready.wait()
            session.add(Resolution(complaint_id=complaint.id, version=version, draft_text="New"))
            try:
                await session.commit()
                return pid, version, "committed"
            except IntegrityError as error:
                await session.rollback()
                assert error.orig.sqlstate == "23505"
                return pid, version, "conflict"

    outcomes = await asyncio.wait_for(asyncio.gather(writer(), writer()), timeout=15)
    assert len({row[0] for row in outcomes}) == 2
    assert [row[1] for row in outcomes] == [2, 2]
    assert sorted(row[2] for row in outcomes) == ["committed", "conflict"]
    async with pg_sessionmaker() as session:
        versions = (
            await session.exec(
                select(Resolution.version)
                .where(Resolution.complaint_id == complaint.id)
                .order_by(Resolution.version)
            )
        ).all()
        assert versions == [1, 2]


async def test_flush_is_invisible_until_commit_on_another_connection(pg_sessionmaker):
    complaint = Complaint(narrative="Transaction visibility fixture")
    async with pg_sessionmaker() as writer, pg_sessionmaker() as reader:
        writer_pid = await writer.scalar(text("SELECT pg_backend_pid()"))
        reader_pid = await reader.scalar(text("SELECT pg_backend_pid()"))
        assert writer_pid != reader_pid
        writer.add(complaint)
        await writer.flush()
        assert await reader.get(Complaint, complaint.id) is None
        await writer.commit()
        # READ COMMITTED gives each statement a fresh snapshot in PostgreSQL.
        assert await reader.get(Complaint, complaint.id) is not None
