"""Bounded operator batches; callers commit transitions and jobs together."""

from __future__ import annotations

import uuid

from sqlalchemy import exists
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.job import Job, JobKind
from app.services.case_transitions import advance_case
from app.services.job_requests import admit
from app.services.jobs import enqueue_job

BATCH_MAX = 500


async def enqueue_batch(
    session: AsyncSession, kind: JobKind, limit: int, *, requested_by: uuid.UUID | None = None
) -> list[Job]:
    if not 1 <= limit <= BATCH_MAX:
        raise ValueError(f"limit must be between 1 and {BATCH_MAX}")
    if kind not in {JobKind.classification, JobKind.resolution}:
        raise ValueError("batch kind must be classification or resolution")
    await admit(session)
    conditions = [
        Complaint.status
        == (
            ComplaintStatus.pending if kind == JobKind.classification else ComplaintStatus.escalated
        ),
        ~exists().where(
            Job.complaint_id == Complaint.id,
            Job.kind == kind,
            Job.case_version == Complaint.case_version,
            Job.input_revision == Complaint.input_revision,
        ),
    ]
    if kind == JobKind.resolution:
        conditions.extend(
            [
                Complaint.classification_outcome != ClassificationState.unavailable,
                Complaint.classification_stale.is_(False),
                Complaint.sentiment.is_not(None),
                Complaint.intent.is_not(None),
                Complaint.urgency.is_not(None),
            ]
        )
    complaints = (
        await session.exec(
            select(Complaint)
            .where(*conditions)
            .order_by(Complaint.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
    ).all()
    # Reject the whole batch visibly if capacity is insufficient. Nothing is
    # published or committed before this check, so there is no partial success.
    await admit(session, len(complaints))
    jobs = []
    for complaint in complaints:
        if kind == JobKind.resolution:
            complaint.status = ComplaintStatus.agent_triggered
            advance_case(complaint)
            session.add(complaint)
        jobs.append(
            await enqueue_job(
                session,
                complaint,
                kind,
                requested_by=requested_by,
                max_attempts=settings.job_max_attempts,
            )
        )
    return jobs
