"""Fast job lifecycle checks; PostgreSQL tests establish concurrency guarantees."""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.database import engine
from app.models.complaint import Complaint, ComplaintStatus
from app.models.job import Job, JobKind, JobStatus
from app.services.jobs import (
    JobCapacityExceeded,
    JobConflict,
    claim_job,
    complete_job,
    enqueue_job,
    fail_job,
    heartbeat_job,
    lock_owned_job,
    request_index_repair,
    reserve_capacity,
    retry_job,
    supersede_job,
)


@pytest_asyncio.fixture
async def sessions(client):
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def seed(sessions, *, max_attempts=3):
    async with sessions() as session:
        case = Complaint(narrative="Synthetic durable job")
        session.add(case)
        job = await enqueue_job(session, case, JobKind.classification, max_attempts=max_attempts)
        await session.commit()
        return case, job


async def test_transition_and_intent_rollback_together(sessions):
    async with sessions() as session:
        case = Complaint(narrative="Rolled back intake")
        session.add(case)
        await enqueue_job(session, case, JobKind.classification)
        await session.rollback()
    async with sessions() as session:
        assert (await session.exec(select(Complaint))).all() == []
        assert (await session.exec(select(Job))).all() == []


async def test_same_intent_is_retained_after_completion_and_new_revision_is_distinct(sessions):
    case, job = await seed(sessions)
    async with sessions() as session:
        owned = await claim_job(session, JobKind.classification)
        complete_job(owned)
        await session.commit()
    async with sessions() as session:
        duplicate = await enqueue_job(session, case, JobKind.classification)
        assert duplicate.id == job.id
        assert duplicate.status == JobStatus.succeeded
        case.case_version += 1
        session.add(case)
        new = await enqueue_job(session, case, JobKind.classification)
        await session.commit()
        assert new.id != job.id
        assert new.input_revision == job.input_revision


async def test_idempotency_key_reuse_for_different_request_conflicts(sessions):
    async with sessions() as session:
        first = Complaint(narrative="Original")
        second = Complaint(narrative="Different case")
        session.add_all([first, second])
        await enqueue_job(session, first, JobKind.classification, idempotency_key="request-1")
        with pytest.raises(JobConflict):
            await enqueue_job(session, second, JobKind.classification, idempotency_key="request-1")
        await session.rollback()


async def test_duplicate_intent_cannot_replace_payload(sessions):
    async with sessions() as session:
        case = Complaint(narrative="Synthetic intent")
        session.add(case)
        await enqueue_job(session, case, JobKind.resolution, payload={"feedback": "first"})
        with pytest.raises(JobConflict):
            await enqueue_job(session, case, JobKind.resolution, payload={"feedback": "second"})


async def test_attempts_backoff_and_final_failure_are_bounded(sessions):
    _, job = await seed(sessions, max_attempts=2)
    async with sessions() as session:
        first = await claim_job(session, JobKind.classification)
        fail_job(first, "providers_unavailable", now=datetime(2026, 1, 1))
        assert first.status == JobStatus.queued
        assert first.next_attempt_at == datetime(2026, 1, 1, 0, 0, 5)
        await session.commit()
    async with sessions() as session:
        second = await claim_job(session, JobKind.classification)
        assert second.id == job.id
        assert second.attempts == second.total_attempts == 2
        fail_job(second, "providers_unavailable")
        assert second.status == JobStatus.failed
        assert second.completed_at is not None
        assert second.lease_token is second.lease_expires_at is None
        await session.commit()
    async with sessions() as session:
        assert await claim_job(session, JobKind.classification) is None


async def test_terminal_errors_fail_without_retry_and_metadata_is_sanitized(sessions):
    await seed(sessions)
    async with sessions() as session:
        job = await claim_job(session, JobKind.classification)
        with pytest.raises(ValueError, match="safe category"):
            fail_job(job, "customer data: unsafe exception message")
        fail_job(job, "invalid_input", retryable=False)
        assert job.status == JobStatus.failed
        assert job.attempts == 1


async def test_expired_final_lease_is_visible_failed_without_another_call(sessions):
    _, original = await seed(sessions, max_attempts=1)
    async with sessions() as session:
        job = await claim_job(session, JobKind.classification)
        job.lease_expires_at = datetime(2000, 1, 1)
        await session.commit()
    async with sessions() as session:
        assert await claim_job(session, JobKind.classification) is None
        await session.commit()
        failed = await session.get(Job, original.id)
        assert failed.status == JobStatus.failed
        assert failed.last_error_code == "lease_exhausted"
        assert failed.total_attempts == 1


async def test_heartbeat_requires_current_live_lease_and_preserves_attempt(sessions):
    await seed(sessions)
    async with sessions() as session:
        job = await claim_job(session, JobKind.classification, lease_seconds=30)
        old_expiry = job.lease_expires_at
        assert not await heartbeat_job(session, job.id, uuid.uuid4())
        assert await heartbeat_job(session, job.id, job.lease_token, lease_seconds=90)
        assert job.lease_expires_at >= old_expiry + timedelta(seconds=60)
        assert job.attempts == 1
        job.lease_expires_at = datetime(2000, 1, 1)
        await session.flush()
        assert not await heartbeat_job(session, job.id, job.lease_token)


async def test_expired_worker_token_cannot_complete_reclaimed_job(sessions):
    await seed(sessions)
    async with sessions() as session:
        job = await claim_job(session, JobKind.classification)
        first_token = job.lease_token
        job.lease_expires_at = datetime(2000, 1, 1)
        await session.commit()
    async with sessions() as session:
        current = await claim_job(session, JobKind.classification)
        assert current.lease_token != first_token
        assert await lock_owned_job(session, current.id, first_token) is None
        complete_job(current)
        await session.commit()
    async with sessions() as session:
        assert await lock_owned_job(session, current.id, first_token) is None
        assert (await session.get(Job, current.id)).status == JobStatus.succeeded


async def test_heartbeat_cannot_revive_lease_expiring_during_renewal(sessions, monkeypatch):
    await seed(sessions)
    async with sessions() as session:
        job = await claim_job(session, JobKind.classification)
        expiry = job.lease_expires_at
        heartbeat_at = job.heartbeat_at
        monkeypatch.setattr(
            "app.services.jobs.database_now",
            AsyncMock(side_effect=[expiry - timedelta(microseconds=1), expiry]),
        )
        assert not await heartbeat_job(session, job.id, job.lease_token)
        assert job.lease_expires_at == expiry
        assert job.heartbeat_at == heartbeat_at


async def test_new_index_repair_uses_configured_attempt_budget(sessions):
    case, _ = await seed(sessions)
    async with sessions() as session:
        repair = await request_index_repair(session, case, max_attempts=1)
        assert repair.kind == JobKind.index
        assert repair.max_attempts == 1


async def test_manual_retry_replay_preserves_lifetime_attempts_and_one_intent(sessions):
    _, original = await seed(sessions, max_attempts=1)
    async with sessions() as session:
        failed = await claim_job(session, JobKind.classification)
        fail_job(failed, "providers_unavailable")
        await session.commit()
    async with sessions() as session:
        retried = await retry_job(
            session, original.id, expected_attempts=1, expected_retry_generation=0
        )
        assert retried.attempts == 0
        assert retried.total_attempts == retried.retry_generation == 1
        await session.commit()
    async with sessions() as session:
        active = await claim_job(session, JobKind.classification)
        assert active.attempts == 1
        assert active.total_attempts == 2
        token = active.lease_token
        await session.commit()
    async with sessions() as session:
        repeated = await retry_job(
            session, original.id, expected_attempts=1, expected_retry_generation=0
        )
        assert repeated.status == JobStatus.running
        assert repeated.lease_token == token
        assert repeated.retry_generation == 1
        assert len((await session.exec(select(Job))).all()) == 1
        with pytest.raises(JobConflict, match="retry identity"):
            await retry_job(session, original.id, expected_attempts=2, expected_retry_generation=0)


async def test_manual_retry_cannot_reopen_reviewed_case(sessions):
    case, original = await seed(sessions, max_attempts=1)
    async with sessions() as session:
        job = await claim_job(session, JobKind.classification)
        fail_job(job, "providers_unavailable")
        case.status = ComplaintStatus.response_approved
        session.add(case)
        await session.commit()
    async with sessions() as session:
        with pytest.raises(JobConflict, match="case changed"):
            await retry_job(session, original.id, expected_attempts=1, expected_retry_generation=0)


async def test_superseded_job_is_retained_and_cannot_be_claimed(sessions):
    await seed(sessions)
    async with sessions() as session:
        job = await claim_job(session, JobKind.classification)
        supersede_job(job)
        await session.commit()
        assert job.status == JobStatus.superseded
        assert job.last_error_code == "stale_case"
        assert await claim_job(session, JobKind.classification) is None


async def test_admission_rejects_before_intake_but_allows_replay_lock_when_full(sessions):
    await seed(sessions)
    async with sessions() as session:
        await reserve_capacity(session, 0, max_pending=1)
        with pytest.raises(JobCapacityExceeded):
            await reserve_capacity(session, 1, max_pending=1)
        assert len((await session.exec(select(Job))).all()) == 1
