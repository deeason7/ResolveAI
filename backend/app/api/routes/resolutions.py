"""Exact saved-draft review; state and immutable review evidence commit together."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.core.deps import get_current_user, require_writer
from app.database import get_session
from app.models.audit_log import AuditLog
from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.job import Job, JobKind, JobStatus
from app.models.resolution import Resolution
from app.models.review_event import ReviewEvent
from app.models.user import User
from app.schemas.resolution import (
    RejectRequest,
    ResolutionQueued,
    ResolutionRead,
    ReviewOutcome,
    ReviewRequest,
)
from app.services.case_transitions import (
    TERMINAL_STATUSES,
    advance_case,
    conflict,
    lock_case,
    review_revision,
)
from app.services.job_requests import admit, fingerprint, replay, request_identity
from app.services.jobs import enqueue_job

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/resolutions", tags=["resolutions"])


def _require_current_classification(complaint: Complaint) -> None:
    if (
        complaint.sentiment is None
        or complaint.intent is None
        or complaint.urgency is None
        or complaint.classification_outcome == ClassificationState.unavailable
        or complaint.classification_stale
    ):
        raise conflict("Complaint needs a current classification before generating a draft")


@router.post(
    "/{complaint_id}/generate",
    response_model=ResolutionQueued,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_resolution(
    complaint_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(require_writer),
    idempotency_key: str | None = Header(default=None, min_length=1, max_length=128),
) -> ResolutionQueued:
    await admit(session)
    key = request_identity(current_user.id, f"resolutions.generate:{complaint_id}", idempotency_key)
    request_fingerprint = fingerprint({"complaint_id": complaint_id})
    existing = await replay(session, key, request_fingerprint)
    if existing is not None:
        return ResolutionQueued(
            complaint_id=complaint_id, job_id=existing.id, status=existing.status
        )
    complaint = await lock_case(session, complaint_id)
    if complaint is None:
        raise HTTPException(404, "Complaint not found")
    _require_current_classification(complaint)
    current_job = (
        await session.exec(
            select(Job).where(
                Job.complaint_id == complaint_id,
                Job.kind == JobKind.resolution,
                Job.case_version == complaint.case_version,
                Job.input_revision == complaint.input_revision,
                Job.status.in_([JobStatus.queued, JobStatus.running, JobStatus.failed]),
            )
        )
    ).first()
    if current_job is not None:
        if current_job.status == JobStatus.failed:
            raise conflict("The draft job failed; inspect and retry that job from Workspace")
        # Classification may already have scheduled a draft while the case is
        # still escalated. Preserve that intent instead of advancing past it.
        raise conflict("Resolution generation is already queued or running")
    if complaint.status in TERMINAL_STATUSES:
        raise conflict("A response has already been approved")
    if complaint.status == ComplaintStatus.agent_triggered:
        raise conflict("Resolution generation already in progress")
    # A failed draft is still a reviewable revision. Require a rejection before
    # replacing it; unavailable inference with no draft can be retried directly.
    latest = (
        await session.exec(
            select(Resolution)
            .where(Resolution.complaint_id == complaint_id)
            .order_by(Resolution.version.desc())
            .limit(1)
        )
    ).first()
    if latest is not None:
        reviewed = (
            await session.exec(select(ReviewEvent).where(ReviewEvent.resolution_id == latest.id))
        ).first()
        if reviewed is None and latest.input_revision == complaint.input_revision:
            raise conflict("A draft is already awaiting review; reject it to regenerate")
    if complaint.status == ComplaintStatus.draft_ready:
        raise conflict("A draft is already awaiting review; reject it to regenerate")
    await admit(session, 1)
    complaint.status = ComplaintStatus.agent_triggered
    advance_case(complaint)
    session.add(complaint)
    job = await enqueue_job(
        session,
        complaint,
        JobKind.resolution,
        idempotency_key=key,
        requested_by=current_user.id,
        request_fingerprint=request_fingerprint,
        max_attempts=settings.job_max_attempts,
    )
    await session.commit()
    return ResolutionQueued(complaint_id=complaint_id, job_id=job.id)


async def _read_revisions(
    session: AsyncSession, complaint_id: uuid.UUID, *, latest_only: bool = False
) -> list[ResolutionRead]:
    # The shared parent lock makes case_version and draft identity one snapshot.
    # Every writer locks the same parent before allocating or reviewing drafts.
    complaint = (
        await session.exec(
            select(Complaint)
            .where(Complaint.id == complaint_id)
            .with_for_update(read=True)
            .execution_options(populate_existing=True)
        )
    ).one_or_none()
    if complaint is None:
        raise HTTPException(404, "Complaint not found")
    query = (
        select(Resolution)
        .where(Resolution.complaint_id == complaint_id)
        .order_by(Resolution.version.desc())
    )
    if latest_only:
        query = query.limit(1)
    rows = (await session.exec(query)).all()
    events = (
        await session.exec(
            select(ReviewEvent).where(ReviewEvent.resolution_id.in_([row.id for row in rows]))
        )
    ).all()
    actions = {event.resolution_id: event.action for event in events}
    return [
        ResolutionRead(
            **row.model_dump(),
            case_version=complaint.case_version,
            review_action=actions.get(row.id),
        )
        for row in rows
    ]


@router.get("/{complaint_id}", response_model=ResolutionRead)
async def get_resolution(
    complaint_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> ResolutionRead:
    rows = await _read_revisions(session, complaint_id, latest_only=True)
    if not rows:
        raise HTTPException(404, "No resolution drafted for this complaint yet")
    return rows[0]


@router.get("/{complaint_id}/revisions", response_model=list[ResolutionRead])
async def list_revisions(
    complaint_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> list[ResolutionRead]:
    return await _read_revisions(session, complaint_id)


def _audit(
    session: AsyncSession,
    current_user: User,
    complaint: Complaint,
    resolution: Resolution,
    event: ReviewEvent,
) -> None:
    session.add(
        AuditLog(
            user_id=current_user.id,
            event=f"resolution_{event.action}",
            metadata_={
                "complaint_id": str(complaint.id),
                "resolution_id": str(resolution.id),
                "version": resolution.version,
                "case_version": complaint.case_version,
                "review_event_id": str(event.id),
                "feedback": event.feedback,
            },
        )
    )


@router.post("/{complaint_id}/approve", response_model=ReviewOutcome)
async def approve_resolution(
    complaint_id: uuid.UUID,
    body: ReviewRequest,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(require_writer),
) -> ReviewOutcome:
    complaint, resolution, event = await review_revision(
        session,
        complaint_id,
        body.resolution_id,
        body.expected_case_version,
        current_user.id,
        action="approved",
    )
    _audit(session, current_user, complaint, resolution, event)
    await session.flush()
    return ReviewOutcome(
        complaint_id=complaint_id,
        resolution_id=resolution.id,
        version=resolution.version,
        action="approved",
        complaint_status=complaint.status.value,
        case_version=complaint.case_version,
    )


@router.post(
    "/{complaint_id}/reject", response_model=ResolutionQueued, status_code=status.HTTP_202_ACCEPTED
)
async def reject_resolution(
    complaint_id: uuid.UUID,
    body: RejectRequest,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(require_writer),
    idempotency_key: str | None = Header(default=None, min_length=1, max_length=128),
) -> ResolutionQueued:
    await admit(session)
    # Exact review identity is a stable retry intent even without a client key.
    review_key = idempotency_key or f"{body.resolution_id}:{body.expected_case_version}"
    key = request_identity(current_user.id, f"resolutions.reject:{complaint_id}", review_key)
    request_fingerprint = fingerprint(body.model_dump())
    existing = await replay(session, key, request_fingerprint)
    if existing is not None:
        return ResolutionQueued(
            complaint_id=complaint_id, job_id=existing.id, status=existing.status
        )
    await admit(session, 1)
    complaint, resolution, event = await review_revision(
        session,
        complaint_id,
        body.resolution_id,
        body.expected_case_version,
        current_user.id,
        action="rejected",
        feedback=body.feedback,
    )
    _require_current_classification(complaint)
    _audit(session, current_user, complaint, resolution, event)
    await session.flush()
    job = await enqueue_job(
        session,
        complaint,
        JobKind.resolution,
        idempotency_key=key,
        requested_by=current_user.id,
        review_event_id=event.id,
        payload={"feedback": body.feedback},
        request_fingerprint=request_fingerprint,
        max_attempts=settings.job_max_attempts,
    )
    await session.commit()
    return ResolutionQueued(complaint_id=complaint_id, job_id=job.id)
