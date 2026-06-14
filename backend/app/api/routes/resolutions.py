"""
Resolution endpoints — the human side of the agent loop:

  POST /resolutions/{complaint_id}/generate   — trigger the agent (202, async)
  GET  /resolutions/{complaint_id}            — latest draft
  GET  /resolutions/{complaint_id}/revisions  — full version history
  POST /resolutions/{complaint_id}/approve    — sign off; complaint -> resolved
  POST /resolutions/{complaint_id}/reject     — feedback -> regenerate (202)

Generation is asynchronous by design: drafting + guardrails is seconds of LLM
time, far past what a request should block on. The trigger routes flip the
complaint to agent_triggered and XADD to the resolution stream; 202 means
"accepted, poll the GET". Approve/reject are the human checkpoint — the agent
never sends anything to a consumer on its own.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.deps import get_current_user, get_redis
from app.database import get_session
from app.models.audit_log import AuditLog
from app.models.complaint import Complaint, ComplaintStatus
from app.models.resolution import GuardrailStatus, Resolution
from app.models.user import User
from app.schemas.resolution import (
    RejectRequest,
    ResolutionQueued,
    ResolutionRead,
    ReviewOutcome,
)
from app.workers.resolution_worker import enqueue_resolution

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/resolutions", tags=["resolutions"])


def _status_value(complaint: Complaint) -> str:
    """Complaint status as its value string.

    EnumString round-trips the column as the enum in both directions, so this is
    always ``.value`` — no isinstance() guard needed.
    """
    return complaint.status.value


async def _complaint_or_404(session: AsyncSession, complaint_id: uuid.UUID) -> Complaint:
    complaint = await session.get(Complaint, complaint_id)
    if complaint is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")
    return complaint


async def _latest_resolution(session: AsyncSession, complaint_id: uuid.UUID) -> Resolution | None:
    stmt = (
        select(Resolution)
        .where(Resolution.complaint_id == complaint_id)
        .order_by(Resolution.version.desc())
        .limit(1)
    )
    return (await session.exec(stmt)).first()


@router.post(
    "/{complaint_id}/generate",
    response_model=ResolutionQueued,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_resolution(
    complaint_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    redis: aioredis.Redis = Depends(get_redis),
    _: User = Depends(get_current_user),
) -> ResolutionQueued:
    """Manually trigger the resolution agent for a complaint.

    The status flips to agent_triggered *in this request* so a double-click (or
    a second analyst) gets a clean 409 instead of a duplicate draft version.
    """
    complaint = await _complaint_or_404(session, complaint_id)
    current = _status_value(complaint)

    if complaint.sentiment is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Complaint is awaiting classification; generate after it's classified",
        )
    if current == ComplaintStatus.agent_triggered.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Resolution generation already in progress",
        )
    if current == ComplaintStatus.draft_ready.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A draft is already awaiting review; reject it to regenerate",
        )
    if current == ComplaintStatus.resolved.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Complaint is already resolved",
        )

    complaint.status = ComplaintStatus.agent_triggered
    complaint.updated_at = datetime.utcnow()
    session.add(complaint)
    # Flush before XADD: if the enqueue raises, the request 500s and the
    # session dependency rolls the status flip back with it.
    await session.flush()
    await enqueue_resolution(redis, complaint_id)
    logger.info("resolution generation queued for %s", complaint_id)
    return ResolutionQueued(complaint_id=complaint_id)


@router.get("/{complaint_id}", response_model=ResolutionRead)
async def get_resolution(
    complaint_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> ResolutionRead:
    """Latest resolution draft for a complaint."""
    await _complaint_or_404(session, complaint_id)
    resolution = await _latest_resolution(session, complaint_id)
    if resolution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No resolution drafted for this complaint yet",
        )
    return ResolutionRead.model_validate(resolution)


@router.get("/{complaint_id}/revisions", response_model=list[ResolutionRead])
async def list_revisions(
    complaint_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> list[ResolutionRead]:
    """Every draft version, newest first. Empty list is a valid 200."""
    await _complaint_or_404(session, complaint_id)
    stmt = (
        select(Resolution)
        .where(Resolution.complaint_id == complaint_id)
        .order_by(Resolution.version.desc())
    )
    rows = (await session.exec(stmt)).all()
    return [ResolutionRead.model_validate(r) for r in rows]


@router.post("/{complaint_id}/approve", response_model=ReviewOutcome)
async def approve_resolution(
    complaint_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> ReviewOutcome:
    """Sign off on the latest draft; the complaint is resolved.

    Only guardrail-passed drafts are one-click approvable — a failed draft
    contains the exact text the guardrails flagged (legal advice, ungrounded
    citations, ...), so the path for those is reject-with-feedback, not an
    override. The approval lands in the audit log with the reviewer's id.
    """
    complaint = await _complaint_or_404(session, complaint_id)
    if _status_value(complaint) == ComplaintStatus.resolved.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Complaint is already resolved"
        )
    resolution = await _latest_resolution(session, complaint_id)
    if resolution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No resolution to approve"
        )
    if resolution.guardrail_status != GuardrailStatus.passed.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Latest draft did not pass guardrails; reject it with feedback instead",
        )

    complaint.status = ComplaintStatus.resolved
    complaint.updated_at = datetime.utcnow()
    session.add(complaint)
    session.add(
        AuditLog(
            user_id=current_user.id,
            event="resolution_approved",
            metadata_={
                "complaint_id": str(complaint_id),
                "resolution_id": str(resolution.id),
                "version": resolution.version,
            },
        )
    )
    logger.info(
        "resolution %s v%d approved for complaint %s",
        resolution.id,
        resolution.version,
        complaint_id,
    )
    return ReviewOutcome(
        complaint_id=complaint_id,
        resolution_id=resolution.id,
        version=resolution.version,
        action="approved",
        complaint_status=ComplaintStatus.resolved.value,
    )


@router.post(
    "/{complaint_id}/reject",
    response_model=ResolutionQueued,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reject_resolution(
    complaint_id: uuid.UUID,
    body: RejectRequest,
    session: AsyncSession = Depends(get_session),
    redis: aioredis.Redis = Depends(get_redis),
    current_user: User = Depends(get_current_user),
) -> ResolutionQueued:
    """Reject the latest draft; the feedback drives a regeneration.

    The reviewer's words travel on the stream message and seed the agent's next
    attempt as a revision of the rejected text — same mechanism the guardrails
    use, so the model revises instead of starting blind.
    """
    complaint = await _complaint_or_404(session, complaint_id)
    if _status_value(complaint) == ComplaintStatus.resolved.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Complaint is already resolved; nothing to reject",
        )
    resolution = await _latest_resolution(session, complaint_id)
    if resolution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No resolution to reject")

    complaint.status = ComplaintStatus.agent_triggered
    complaint.updated_at = datetime.utcnow()
    session.add(complaint)
    session.add(
        AuditLog(
            user_id=current_user.id,
            event="resolution_rejected",
            metadata_={
                "complaint_id": str(complaint_id),
                "resolution_id": str(resolution.id),
                "version": resolution.version,
                "feedback": body.feedback,
            },
        )
    )
    await session.flush()
    await enqueue_resolution(redis, complaint_id, feedback=body.feedback)
    logger.info(
        "resolution %s v%d rejected for complaint %s; regeneration queued",
        resolution.id,
        resolution.version,
        complaint_id,
    )
    return ResolutionQueued(complaint_id=complaint_id)
