"""Versioned case transitions. Callers commit state and evidence together."""

import hashlib
import uuid
from datetime import datetime

from fastapi import HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.complaint import Complaint, ComplaintStatus
from app.models.resolution import GuardrailStatus, Resolution
from app.models.review_event import ReviewEvent

TERMINAL_STATUSES = {ComplaintStatus.response_approved, ComplaintStatus.resolved}


async def lock_case(session: AsyncSession, complaint_id: uuid.UUID) -> Complaint | None:
    return (
        await session.exec(
            select(Complaint)
            .where(Complaint.id == complaint_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).one_or_none()


def advance_case(complaint: Complaint, *, inputs_changed: bool = False) -> None:
    complaint.case_version += 1
    if inputs_changed:
        complaint.input_revision += 1
    complaint.updated_at = datetime.utcnow()


def conflict(reason: str) -> HTTPException:
    return HTTPException(
        409, f"{reason}. Reload the current draft and review it before submitting again."
    )


async def review_revision(
    session: AsyncSession,
    complaint_id: uuid.UUID,
    resolution_id: uuid.UUID,
    expected_case_version: int,
    user_id: uuid.UUID,
    *,
    action: str,
    feedback: str | None = None,
) -> tuple[Complaint, Resolution, ReviewEvent]:
    complaint = await lock_case(session, complaint_id)
    if complaint is None:
        raise HTTPException(404, "Complaint not found")
    if complaint.case_version != expected_case_version:
        raise conflict("The case changed since this draft was displayed")
    if complaint.status in TERMINAL_STATUSES:
        raise conflict("A response has already been approved")
    resolution = await session.get(Resolution, resolution_id)
    if resolution is None or resolution.complaint_id != complaint_id:
        raise conflict("The selected draft does not belong to this case")
    latest = (
        await session.exec(
            select(Resolution.id)
            .where(Resolution.complaint_id == complaint_id)
            .order_by(Resolution.version.desc())
            .limit(1)
        )
    ).first()
    if latest != resolution_id or resolution.input_revision != complaint.input_revision:
        raise conflict("The selected draft is no longer current")
    if (
        await session.exec(select(ReviewEvent.id).where(ReviewEvent.resolution_id == resolution_id))
    ).first() is not None:
        raise conflict("This draft has already been reviewed")
    if complaint.status not in {ComplaintStatus.draft_ready, ComplaintStatus.needs_review}:
        raise conflict("The case is not awaiting review")
    if action == "approved" and resolution.guardrail_status != GuardrailStatus.passed:
        raise conflict("The draft did not pass guardrails; reject it with feedback")
    if action not in {"approved", "rejected"}:
        raise ValueError("Unsupported review action")
    advance_case(complaint)
    complaint.status = (
        ComplaintStatus.response_approved
        if action == "approved"
        else ComplaintStatus.agent_triggered
    )
    event = ReviewEvent(
        complaint_id=complaint_id,
        resolution_id=resolution_id,
        user_id=user_id,
        action=action,
        expected_case_version=expected_case_version,
        case_version=complaint.case_version,
        input_revision=complaint.input_revision,
        draft_sha256=hashlib.sha256(resolution.draft_text.encode("utf-8")).hexdigest(),
        feedback=feedback,
    )
    session.add(complaint)
    session.add(event)
    return complaint, resolution, event


def message_versions(fields: dict[str, str]) -> tuple[int, int] | None:
    """Legacy messages cannot prove which case snapshot requested inference."""
    try:
        values = (int(fields["expected_case_version"]), int(fields["input_revision"]))
    except (KeyError, TypeError, ValueError):
        return None
    return values if all(value >= 1 for value in values) else None
