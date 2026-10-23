"""Authorized job inspection and generation-fenced manual retries."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.deps import require_writer
from app.database import get_session
from app.models.job import Job, JobKind, JobStatus
from app.models.user import User
from app.schemas.job import JobList, JobRead, RetryRequest
from app.services.job_requests import admit
from app.services.jobs import JobConflict, retry_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/", response_model=JobList)
async def list_jobs(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(require_writer),
    status: JobStatus = Query(default=JobStatus.failed),
    kind: JobKind | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> JobList:
    conditions = [Job.status == status]
    if kind is not None:
        conditions.append(Job.kind == kind)
    total = (await session.exec(select(func.count(Job.id)).where(*conditions))).one()
    rows = (
        await session.exec(
            select(Job)
            .where(*conditions)
            .order_by(Job.updated_at.desc(), Job.id)
            .offset(offset)
            .limit(limit)
        )
    ).all()
    return JobList(
        items=[JobRead.model_validate(job) for job in rows], total=total, limit=limit, offset=offset
    )


@router.get("/{job_id}", response_model=JobRead)
async def get_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(require_writer),
) -> JobRead:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JobRead.model_validate(job)


@router.post("/{job_id}/retry", response_model=JobRead)
async def retry_failed_job(
    job_id: uuid.UUID,
    body: RetryRequest,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(require_writer),
) -> JobRead:
    await admit(session)
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status == JobStatus.failed and job.retry_generation == body.expected_retry_generation:
        await admit(session, 1)
    try:
        job = await retry_job(session, job_id, **body.model_dump())
    except JobConflict as exc:
        raise HTTPException(409, str(exc)) from None
    await session.commit()
    return JobRead.model_validate(job)
