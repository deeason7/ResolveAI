"""Transactional jobs. Callers own commits, and external calls hold no locks.

When a transaction needs both rows, lock the case before its job. Admission takes
its advisory lock before either. Claims and heartbeats lock only jobs. A worker
must lock_owned_job after inference and persist result + completion together.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import DateTime, and_, func, or_, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.complaint import Complaint
from app.models.job import Job, JobKind, JobStatus
from app.services.case_transitions import TERMINAL_STATUSES, lock_case

ACTIVE_STATUSES = (JobStatus.queued, JobStatus.running)
_ADMISSION_LOCK = 714_077_001
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class JobConflict(ValueError):
    """A request identity was reused, or the requested job state is stale."""


class JobCapacityExceeded(ValueError):
    """Admission failed before accepting any new business transition."""


def _database_clock(session: AsyncSession):
    # PostgreSQL CURRENT_TIMESTAMP is fixed at transaction start, which is unsafe
    # after waiting for a row lock. Lease checks require the actual database time.
    if session.get_bind().dialect.name == "postgresql":
        return func.timezone("UTC", func.clock_timestamp())
    # Match SQLite's six-digit datetime storage so equal timestamps compare equal.
    return func.strftime("%Y-%m-%d %H:%M:%f000", "now", type_=DateTime())


async def database_now(session: AsyncSession) -> datetime:
    value = await session.scalar(select(_database_clock(session)))
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value


async def reserve_capacity(session: AsyncSession, count: int, *, max_pending: int) -> None:
    """Serialize external admission; accepted follow-up jobs bypass this ceiling.

    This is a backlog admission threshold, not a deletion limit. In-progress
    workflows may add their bounded follow-up work even when intake is closed.
    SQLite supports unit behavior only; production concurrency uses PostgreSQL.
    """
    if count < 0 or max_pending < 1:
        raise ValueError("Invalid job admission limit")
    if session.get_bind().dialect.name == "postgresql":
        await session.exec(
            text("SELECT pg_advisory_xact_lock(:key)"), params={"key": _ADMISSION_LOCK}
        )
    if count == 0:
        return
    pending = await session.scalar(
        select(func.count(Job.id)).where(Job.status.in_(ACTIVE_STATUSES))
    )
    if pending + count > max_pending:
        raise JobCapacityExceeded("The processing backlog is full; try again later")


async def enqueue_job(
    session: AsyncSession,
    complaint: Complaint,
    kind: JobKind | str,
    *,
    idempotency_key: str | None = None,
    requested_by: uuid.UUID | None = None,
    review_event_id: uuid.UUID | None = None,
    payload: dict | None = None,
    request_fingerprint: str | None = None,
    max_attempts: int = 3,
) -> Job:
    """Insert intent alongside business state, or return the existing same intent."""
    kind = JobKind(kind)
    key = (
        idempotency_key
        or f"{kind.value}:{complaint.id}:{complaint.case_version}:{complaint.input_revision}"
    )
    if not key or len(key) > 255 or not 1 <= max_attempts <= 20:
        raise ValueError("Invalid job identity or attempt budget")
    # Flush the case/review event first so both foreign keys are present in this
    # transaction. ON CONFLICT handles two independent writers without rollback.
    await session.flush()
    job = Job(
        complaint_id=complaint.id,
        kind=kind,
        case_version=complaint.case_version,
        input_revision=complaint.input_revision,
        idempotency_key=key,
        requested_by=requested_by,
        review_event_id=review_event_id,
        payload=payload or {},
        request_fingerprint=request_fingerprint,
        max_attempts=max_attempts,
        next_attempt_at=await database_now(session),
    )
    insert = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
    statement = insert(Job).values(**job.model_dump()).on_conflict_do_nothing().returning(Job.id)
    inserted_id = (await session.exec(statement)).scalar_one_or_none()
    if inserted_id is not None:
        return await session.get(Job, inserted_id)
    matches = (
        await session.exec(
            select(Job).where(
                or_(
                    Job.idempotency_key == key,
                    and_(
                        Job.kind == kind,
                        Job.complaint_id == complaint.id,
                        Job.case_version == complaint.case_version,
                        Job.input_revision == complaint.input_revision,
                    ),
                )
            )
        )
    ).all()
    if len(matches) != 1:
        raise JobConflict("The idempotency key belongs to another job")
    existing = matches[0]
    if (
        existing.kind != kind
        or existing.complaint_id != complaint.id
        or existing.case_version != complaint.case_version
        or existing.input_revision != complaint.input_revision
        or existing.request_fingerprint != request_fingerprint
        or existing.payload != (payload or {})
        or existing.review_event_id != review_event_id
    ):
        raise JobConflict("The idempotency key belongs to a different request")
    return existing


def _clear_lease(job: Job) -> None:
    job.lease_token = None
    job.lease_expires_at = None


async def claim_job(
    session: AsyncSession, kind: JobKind | str, *, lease_seconds: int = 120
) -> Job | None:
    """Claim one available job; expired final attempts become visible failures."""
    if lease_seconds < 1:
        raise ValueError("Lease duration must be positive")
    clock = _database_clock(session)
    # Bound maintenance per transaction when many abandoned jobs exhausted their
    # budgets. The next poll continues scanning without holding unbounded locks.
    for _ in range(100):
        job = (
            await session.exec(
                select(Job)
                .where(
                    Job.kind == JobKind(kind),
                    or_(
                        and_(Job.status == JobStatus.queued, Job.next_attempt_at <= clock),
                        and_(Job.status == JobStatus.running, Job.lease_expires_at <= clock),
                    ),
                )
                .order_by(Job.next_attempt_at, Job.created_at, Job.id)
                .with_for_update(skip_locked=True)
                .limit(1)
                .execution_options(populate_existing=True)
            )
        ).first()
        if job is None:
            return None
        now = await database_now(session)
        if job.attempts >= job.max_attempts:
            job.status = JobStatus.failed
            job.last_error_code = "lease_exhausted"
            job.completed_at = job.updated_at = now
            _clear_lease(job)
            await session.flush()
            continue
        job.status = JobStatus.running
        job.attempts += 1
        job.total_attempts += 1
        job.lease_token = uuid.uuid4()
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        job.heartbeat_at = job.updated_at = now
        job.completed_at = None
        await session.flush()
        return job
    return None


async def lock_owned_job(
    session: AsyncSession, job_id: uuid.UUID, lease_token: uuid.UUID
) -> Job | None:
    job = (
        await session.exec(
            select(Job)
            .where(Job.id == job_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).first()
    # Fetch time after obtaining the lock: a wait can outlast the original lease.
    now = await database_now(session)
    if (
        job is None
        or job.status != JobStatus.running
        or job.lease_token != lease_token
        or job.lease_expires_at is None
        or job.lease_expires_at <= now
    ):
        return None
    return job


async def heartbeat_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    *,
    lease_seconds: int = 120,
) -> bool:
    if lease_seconds < 1:
        raise ValueError("Lease duration must be positive")
    job = await lock_owned_job(session, job_id, lease_token)
    if job is None:
        return False
    now = await database_now(session)
    # The database round trip after ownership validation can itself span expiry.
    # Never resurrect an expired lease, even while its row remains locked.
    if job.lease_expires_at <= now:
        return False
    job.heartbeat_at = job.updated_at = now
    job.lease_expires_at = now + timedelta(seconds=lease_seconds)
    await session.flush()
    return True


def complete_job(job: Job, *, now: datetime | None = None) -> None:
    job.status = JobStatus.succeeded
    job.completed_at = job.updated_at = now or datetime.utcnow()
    job.last_error_code = None
    _clear_lease(job)


def supersede_job(job: Job, error_code: str = "stale_case", *, now: datetime | None = None) -> None:
    if not _SAFE_CODE.fullmatch(error_code):
        raise ValueError("Job errors must use a safe category")
    job.status = JobStatus.superseded
    job.last_error_code = error_code
    job.completed_at = job.updated_at = now or datetime.utcnow()
    _clear_lease(job)


def fail_job(
    job: Job, error_code: str, *, retryable: bool = True, now: datetime | None = None
) -> None:
    if not _SAFE_CODE.fullmatch(error_code):
        raise ValueError("Job errors must use a safe category")
    now = now or datetime.utcnow()
    job.last_error_code = error_code
    job.updated_at = now
    _clear_lease(job)
    if retryable and job.attempts < job.max_attempts:
        job.status = JobStatus.queued
        job.next_attempt_at = now + timedelta(seconds=min(300, 5 * 2 ** (job.attempts - 1)))
        job.completed_at = None
    else:
        job.status = JobStatus.failed
        job.completed_at = now


async def retry_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    *,
    expected_attempts: int,
    expected_retry_generation: int,
) -> Job:
    """Give a failed intent a fresh bounded budget, with a replay-safe generation."""
    candidate = await session.get(Job, job_id)
    if candidate is None:
        raise JobConflict("Job not found")
    complaint = await lock_case(session, candidate.complaint_id)
    job = (
        await session.exec(
            select(Job)
            .where(Job.id == job_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).one()
    if expected_retry_generation < 0 or expected_attempts < 1:
        raise JobConflict("Invalid retry identity")
    if job.retry_generation == expected_retry_generation + 1:
        if job.last_retry_attempts != expected_attempts:
            raise JobConflict("The retry identity does not match the accepted request")
        return job
    if (
        job.retry_generation != expected_retry_generation
        or job.attempts != expected_attempts
        or job.status != JobStatus.failed
    ):
        raise JobConflict("The job changed; reload its current state before retrying")
    if complaint is None or complaint.input_revision != job.input_revision:
        raise JobConflict("The job inputs are no longer current")
    # Index jobs are derivatives of persisted state and may finish after review.
    if job.kind != JobKind.index and (
        complaint.case_version != job.case_version or complaint.status in TERMINAL_STATUSES
    ):
        raise JobConflict("The case changed; this job cannot be retried")
    job.status = JobStatus.queued
    job.last_retry_attempts = job.attempts
    job.attempts = 0
    job.retry_generation += 1
    job.next_attempt_at = job.updated_at = await database_now(session)
    job.completed_at = None
    job.last_error_code = None
    _clear_lease(job)
    await session.flush()
    return job


async def request_index_repair(
    session: AsyncSession, complaint: Complaint, *, max_attempts: int = 3
) -> Job:
    """Schedule a current snapshot after an obsolete external write finishes.

    Caller holds the case lock. Reuse the current derivative intent and preserve
    lifetime attempts; invalidating an active lease ensures it cannot mark this
    repair complete. Same-input late writes are equivalent and need no new repair.
    """
    job = (
        await session.exec(
            select(Job)
            .where(
                Job.complaint_id == complaint.id,
                Job.kind == JobKind.index,
                Job.input_revision == complaint.input_revision,
            )
            .order_by(Job.case_version.desc())
            .limit(1)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).first()
    if job is None:
        return await enqueue_job(session, complaint, JobKind.index, max_attempts=max_attempts)
    if job.status == JobStatus.queued:
        return job
    job.status = JobStatus.queued
    job.attempts = 0
    job.retry_generation += 1
    job.last_retry_attempts = None
    job.next_attempt_at = job.updated_at = await database_now(session)
    job.completed_at = None
    job.last_error_code = "index_repair_requested"
    _clear_lease(job)
    await session.flush()
    return job
