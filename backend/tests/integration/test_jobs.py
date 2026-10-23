"""Durable intent, independent claims, fencing and admission on PostgreSQL."""

import asyncio
from datetime import datetime

import pytest
from sqlalchemy import func, insert, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.complaint import Complaint, ComplaintStatus
from app.models.job import Job, JobKind, JobStatus
from app.models.resolution import Resolution
from app.services.case_transitions import lock_case
from app.services.jobs import (
    JobCapacityExceeded,
    claim_job,
    complete_job,
    enqueue_job,
    fail_job,
    heartbeat_job,
    lock_owned_job,
    reserve_capacity,
    retry_job,
)

pytestmark = pytest.mark.postgres


async def seed(sessions, *, max_attempts=3):
    async with sessions() as session:
        case = Complaint(narrative="Synthetic job fixture")
        session.add(case)
        job = await enqueue_job(session, case, JobKind.classification, max_attempts=max_attempts)
        await session.commit()
        return case, job


async def expire(sessions, job_id):
    async with sessions() as session:
        job = await session.get(Job, job_id)
        job.lease_expires_at = datetime(2000, 1, 1)
        await session.commit()


async def test_uncommitted_intake_is_invisible_and_rollback_removes_job(pg_sessionmaker):
    async with pg_sessionmaker() as writer, pg_sessionmaker() as reader:
        case = Complaint(narrative="Uncommitted intake")
        writer.add(case)
        job = await enqueue_job(writer, case, JobKind.classification)
        assert await reader.get(Complaint, case.id) is None
        assert await claim_job(reader, JobKind.classification) is None
        job_id, case_id = job.id, case.id
        await writer.rollback()
        assert await reader.get(Job, job_id) is None
        assert await reader.get(Complaint, case_id) is None


@pytest.mark.parametrize("count", [1, 2])
async def test_skip_locked_claims_have_one_lease_per_job(pg_sessionmaker, count):
    for _ in range(count):
        await seed(pg_sessionmaker)
    started, claimed = asyncio.Barrier(2), asyncio.Barrier(2)

    async def contender():
        async with pg_sessionmaker() as session:
            pid = await session.scalar(text("SELECT pg_backend_pid()"))
            await started.wait()
            job = await claim_job(session, JobKind.classification)
            # Keep the first claim uncommitted until the other connection has
            # tried. A blocking claim (instead of SKIP LOCKED) deadlocks here.
            await claimed.wait()
            await session.commit()
            return pid, (job.id, job.lease_token) if job else None

    results = await asyncio.wait_for(asyncio.gather(contender(), contender()), timeout=10)
    assert len({pid for pid, _ in results}) == 2
    claims = [claim for _, claim in results if claim]
    assert len(claims) == count
    assert len({claim[0] for claim in claims}) == count
    assert len({claim[1] for claim in claims}) == count


async def test_reclaimed_result_fences_late_owner_and_commit_survives_crash(pg_sessionmaker):
    case, job = await seed(pg_sessionmaker)
    async with pg_sessionmaker() as old:
        first = await claim_job(old, JobKind.classification)
        old_token = first.lease_token
        await old.commit()
    await expire(pg_sessionmaker, job.id)
    async with pg_sessionmaker() as replacement:
        second = await claim_job(replacement, JobKind.classification)
        new_token = second.lease_token
        assert new_token != old_token
        await replacement.commit()
    async with pg_sessionmaker() as writer:
        current = await lock_case(writer, case.id)
        owned = await lock_owned_job(writer, job.id, new_token)
        assert owned is not None
        writer.add(Resolution(complaint_id=case.id, draft_text="Replacement's only result"))
        current.status = ComplaintStatus.draft_ready
        current.case_version += 1
        complete_job(owned)
        await writer.commit()
    # Simulate return from an earlier slow inference after the replacement
    # committed, followed by a worker crash before any in-memory acknowledgement.
    async with pg_sessionmaker() as late:
        await lock_case(late, case.id)
        assert await lock_owned_job(late, job.id, old_token) is None
        assert not await heartbeat_job(late, job.id, old_token)
        assert await claim_job(late, JobKind.classification) is None
        assert len((await late.exec(select(Resolution))).all()) == 1
        assert (await late.get(Job, job.id)).status == JobStatus.succeeded


async def test_result_followup_and_completion_rollback_together(pg_sessionmaker):
    case, job = await seed(pg_sessionmaker)
    async with pg_sessionmaker() as session:
        claimed = await claim_job(session, JobKind.classification)
        token = claimed.lease_token
        await session.commit()
    async with pg_sessionmaker() as writer:
        current = await lock_case(writer, case.id)
        owned = await lock_owned_job(writer, job.id, token)
        current.status = ComplaintStatus.escalated
        current.case_version += 1
        writer.add(Resolution(complaint_id=case.id, draft_text="Rolled back result"))
        await enqueue_job(writer, current, JobKind.resolution)
        complete_job(owned)
        await writer.flush()
        await writer.rollback()
    async with pg_sessionmaker() as reader:
        assert (await reader.get(Complaint, case.id)).case_version == 1
        assert (await reader.exec(select(Resolution))).all() == []
        assert len((await reader.exec(select(Job))).all()) == 1
        assert (await reader.get(Job, job.id)).status == JobStatus.running
    await expire(pg_sessionmaker, job.id)
    async with pg_sessionmaker() as replacement:
        recovered = await claim_job(replacement, JobKind.classification)
        assert recovered.id == job.id
        assert recovered.lease_token != token
        assert recovered.attempts == 2


async def test_heartbeat_extends_lease_without_incrementing_attempt(pg_sessionmaker):
    _, job = await seed(pg_sessionmaker)
    async with pg_sessionmaker() as session:
        claimed = await claim_job(session, JobKind.classification, lease_seconds=30)
        token, previous_expiry = claimed.lease_token, claimed.lease_expires_at
        await session.commit()
    async with pg_sessionmaker() as heartbeat:
        assert await heartbeat_job(heartbeat, job.id, token, lease_seconds=120)
        await heartbeat.commit()
    async with pg_sessionmaker() as reader:
        current = await reader.get(Job, job.id)
        assert current.lease_expires_at > previous_expiry
        assert current.attempts == current.total_attempts == 1
        assert await claim_job(reader, JobKind.classification) is None


async def test_repeated_crashes_exhaust_budget_without_silent_drop(pg_sessionmaker):
    _, job = await seed(pg_sessionmaker, max_attempts=2)
    for attempt in (1, 2):
        async with pg_sessionmaker() as session:
            claimed = await claim_job(session, JobKind.classification)
            assert claimed.attempts == attempt
            await session.commit()
        await expire(pg_sessionmaker, job.id)
    async with pg_sessionmaker() as session:
        assert await claim_job(session, JobKind.classification) is None
        await session.commit()
    async with pg_sessionmaker() as reader:
        failed = await reader.get(Job, job.id)
        assert failed.status == JobStatus.failed
        assert failed.last_error_code == "lease_exhausted"
        assert failed.attempts == failed.total_attempts == 2
        assert failed.completed_at is not None


async def test_concurrent_enqueue_deduplicates_one_business_intent(pg_sessionmaker):
    async with pg_sessionmaker() as session:
        case = Complaint(narrative="Concurrent durable intake")
        session.add(case)
        await session.commit()
    ready = asyncio.Barrier(2)

    async def producer():
        async with pg_sessionmaker() as session:
            await ready.wait()
            job = await enqueue_job(session, case, JobKind.classification)
            await session.commit()
            return job.id

    ids = await asyncio.wait_for(asyncio.gather(producer(), producer()), timeout=10)
    assert ids[0] == ids[1]
    async with pg_sessionmaker() as reader:
        assert await reader.scalar(select(func.count(Job.id))) == 1


async def test_concurrent_manual_retry_is_one_budget_and_one_identity(pg_sessionmaker):
    _, job = await seed(pg_sessionmaker, max_attempts=1)
    async with pg_sessionmaker() as session:
        claimed = await claim_job(session, JobKind.classification)
        fail_job(claimed, "providers_unavailable")
        await session.commit()
    ready = asyncio.Barrier(2)

    async def retry():
        async with pg_sessionmaker() as session:
            await ready.wait()
            result = await retry_job(
                session, job.id, expected_attempts=1, expected_retry_generation=0
            )
            await session.commit()
            return result.id, result.retry_generation, result.total_attempts

    results = await asyncio.wait_for(asyncio.gather(retry(), retry()), timeout=10)
    assert results == [(job.id, 1, 1), (job.id, 1, 1)]


async def test_concurrent_admission_rejects_overflow_before_creating_case(pg_sessionmaker):
    ready = asyncio.Barrier(2)

    async def intake():
        async with pg_sessionmaker() as session:
            await ready.wait()
            try:
                await reserve_capacity(session, 1, max_pending=1)
                case = Complaint(narrative="Bounded synthetic intake")
                session.add(case)
                await enqueue_job(session, case, JobKind.classification)
                await session.commit()
                return "accepted"
            except JobCapacityExceeded:
                await session.rollback()
                return "full"

    outcomes = await asyncio.wait_for(asyncio.gather(intake(), intake()), timeout=10)
    assert sorted(outcomes) == ["accepted", "full"]
    async with pg_sessionmaker() as reader:
        assert await reader.scalar(select(func.count(Job.id))) == 1
        assert await reader.scalar(select(func.count(Complaint.id))) == 1


async def test_backlog_above_old_stream_limit_retains_every_intent(pg_engine, pg_sessionmaker):
    case, first = await seed(pg_sessionmaker)
    # Bulk synthetic historical intents exceed the previous stream's MAXLEN.
    # Database retention must have no acknowledgement or count-based trimming.
    rows = [
        Job(
            complaint_id=case.id,
            kind=JobKind.classification,
            case_version=version,
            input_revision=1,
            idempotency_key=f"backlog:{version}",
        ).model_dump()
        for version in range(2, 10_006)
    ]
    async with pg_engine.begin() as connection:
        await connection.execute(insert(Job.__table__), rows)
    async with pg_sessionmaker() as session:
        assert await session.scalar(select(func.count(Job.id))) == 10_005
        assert await session.get(Job, first.id) is not None
        assert (
            await session.scalar(select(func.count(Job.id)).where(Job.status == JobStatus.queued))
            == 10_005
        )


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE jobs SET status = 'lost'",
        "UPDATE jobs SET kind = 'arbitrary'",
        "UPDATE jobs SET attempts = max_attempts + 1",
        "UPDATE jobs SET status = 'running'",
        "UPDATE jobs SET input_revision = 0",
    ],
)
async def test_job_constraints_reject_invalid_persistent_state(pg_sessionmaker, statement):
    await seed(pg_sessionmaker)
    async with pg_sessionmaker() as session:
        with pytest.raises(IntegrityError) as caught:
            await session.exec(text(statement))
        assert caught.value.orig.sqlstate == "23514"
        await session.rollback()


async def test_additive_upgrade_preserves_cases_without_fabricating_legacy_jobs(postgres_database):
    await postgres_database.upgrade("e6b4a9d2c705")
    sessions = async_sessionmaker(
        postgres_database.engine, class_=AsyncSession, expire_on_commit=False
    )
    async with sessions() as session:
        case = Complaint(narrative="Legacy pending work", case_version=3, input_revision=2)
        session.add(case)
        await session.commit()
    await postgres_database.upgrade()
    async with sessions() as reader:
        restored = await reader.get(Complaint, case.id)
        assert (restored.case_version, restored.input_revision) == (3, 2)
        assert (await reader.exec(select(Job))).all() == []


async def test_downgrade_refuses_to_erase_job_history(postgres_database, pg_sessionmaker):
    _, job = await seed(pg_sessionmaker)
    with pytest.raises(pytest.fail.Exception, match="Refusing to remove durable job history"):
        await postgres_database.downgrade("e6b4a9d2c705")
    async with pg_sessionmaker() as session:
        assert await session.get(Job, job.id) is not None
        assert (
            await session.scalar(text("SELECT version_num FROM alembic_version")) == "f7c5b0e3d816"
        )
