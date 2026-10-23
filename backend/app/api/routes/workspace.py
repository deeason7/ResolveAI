"""Pipeline counts and bounded transactional operator batches."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import settings
from app.core.deps import get_current_user, require_writer
from app.database import get_session
from app.middleware.rate_limit import limiter
from app.models.complaint import ClassificationState, Complaint, ComplaintStatus
from app.models.job import Job, JobKind, JobStatus
from app.models.user import User
from app.schemas.job import JobQueueInfo
from app.schemas.workspace import EnqueueResult, WorkspaceBoard
from app.services.case_transitions import TERMINAL_STATUSES
from app.services.job_batches import BATCH_MAX, enqueue_batch

router = APIRouter(prefix="/workspace", tags=["workspace"])
_ENQUEUE_RATE_LIMIT = "10/minute"


async def job_health(session: AsyncSession) -> list[JobQueueInfo]:
    rows = (
        await session.exec(
            select(Job.kind, Job.status, func.count(Job.id)).group_by(Job.kind, Job.status)
        )
    ).all()
    counts = {(kind, state): count for kind, state, count in rows}
    oldest = dict(
        (
            await session.exec(
                select(Job.kind, func.min(Job.created_at))
                .where(Job.status.in_([JobStatus.queued, JobStatus.running]))
                .group_by(Job.kind)
            )
        ).all()
    )
    now = datetime.utcnow()
    return [
        JobQueueInfo(
            kind=kind,
            **{state.value: counts.get((kind, state), 0) for state in JobStatus},
            oldest_pending_age_seconds=max(0.0, (now - oldest[kind]).total_seconds())
            if kind in oldest
            else None,
        )
        for kind in JobKind
    ]


@router.get("/board", response_model=WorkspaceBoard)
async def board(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> WorkspaceBoard:
    rows = (
        await session.exec(
            select(Complaint.status, func.count(Complaint.id)).group_by(Complaint.status)
        )
    ).all()
    counts = {state.value: count for state, count in rows}
    unavailable = (
        await session.exec(
            select(func.count(Complaint.id)).where(
                Complaint.classification_outcome == ClassificationState.unavailable,
                Complaint.status.not_in(TERMINAL_STATUSES),
            )
        )
    ).one()
    return WorkspaceBoard(
        **{state.value: counts.get(state.value, 0) for state in ComplaintStatus},
        total=sum(counts.values()),
        classification_unavailable=unavailable,
        jobs=await job_health(session),
        admission_limit=settings.job_max_pending,
    )


@router.post("/enqueue/classification", response_model=EnqueueResult)
@limiter.limit(_ENQUEUE_RATE_LIMIT)
async def enqueue_classification(
    request: Request,
    limit: int = Query(default=50, ge=1, le=BATCH_MAX),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(require_writer),
) -> EnqueueResult:
    jobs = await enqueue_batch(session, JobKind.classification, limit, requested_by=current_user.id)
    await session.commit()
    return EnqueueResult(enqueued=len(jobs), kind=JobKind.classification)


@router.post("/enqueue/resolution", response_model=EnqueueResult)
@limiter.limit(_ENQUEUE_RATE_LIMIT)
async def enqueue_resolution_batch(
    request: Request,
    limit: int = Query(default=50, ge=1, le=BATCH_MAX),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(require_writer),
) -> EnqueueResult:
    jobs = await enqueue_batch(session, JobKind.resolution, limit, requested_by=current_user.id)
    await session.commit()
    return EnqueueResult(enqueued=len(jobs), kind=JobKind.resolution)
