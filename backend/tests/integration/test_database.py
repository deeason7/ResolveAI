"""Migrations, relational constraints, and controlled independent transactions."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import func, inspect, text
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
        "users",
    }
    assert revision == script.get_current_head()


async def test_previous_revision_upgrades_without_losing_linked_rows(postgres_database):
    await postgres_database.upgrade("d4f8c2b7e1a9")
    sessions = async_sessionmaker(
        postgres_database.engine, class_=AsyncSession, expire_on_commit=False
    )
    complaint = Complaint(narrative="Synthetic reporting dispute", cfpb_complaint_id="migration-1")
    user = User(email="migration@example.test", full_name="Test reviewer", hashed_password="unused")
    async with sessions() as session:
        session.add_all([complaint, user])
        await session.flush()
        label = ComplaintLabel(
            complaint_id=complaint.id,
            label_source="synthetic-test",
            sentiment="negative",
            intent="dispute_resolution",
            urgency=4,
            reasoning="Fixture, not model output",
        )
        draft = Resolution(
            complaint_id=complaint.id,
            draft_text="Synthetic draft retained through upgrade",
            guardrail_violations=[{"code": "fixture"}],
        )
        audit = AuditLog(user_id=user.id, event="migration_fixture")
        session.add_all([label, draft, audit])
        await session.commit()

    await postgres_database.upgrade()

    async with sessions() as session:
        restored = await session.get(Resolution, draft.id)
        assert restored.complaint_id == complaint.id
        assert restored.draft_text == draft.draft_text
        assert restored.guardrail_violations == [{"code": "fixture"}]
        assert (await session.get(ComplaintLabel, label.id)).complaint_id == complaint.id
        assert (await session.get(AuditLog, audit.id)).user_id == user.id
        session.add(Resolution(complaint_id=complaint.id, version=1, draft_text="Duplicate"))
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
