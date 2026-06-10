"""
API contracts for the resolution endpoints.

Same separation-of-schemas discipline as complaints: the ORM row never leaves
the service boundary. ``ResolutionRead`` re-validates the persisted
``guardrail_violations`` dicts back into ``GuardrailViolation`` models, so API
consumers get the typed shape, not whatever the jsonb column happens to hold.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.resolution import GuardrailStatus
from app.schemas.guardrails import GuardrailViolation


class ResolutionRead(BaseModel):
    """One resolution draft, as returned by GET endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    complaint_id: uuid.UUID
    version: int
    draft_text: str
    guardrail_status: GuardrailStatus
    guardrail_notes: str | None = None
    guardrail_violations: list[GuardrailViolation] | None = None
    reasoning_summary: str | None = None
    created_at: datetime
    updated_at: datetime


class ResolutionQueued(BaseModel):
    """202 body for the async trigger endpoints (generate, reject)."""

    complaint_id: uuid.UUID
    status: Literal["queued"] = "queued"


class RejectRequest(BaseModel):
    """Human reviewer's rejection — feedback is fed to the regeneration prompt."""

    feedback: str = Field(
        min_length=10,
        max_length=2000,
        description="What's wrong with the draft and what the next one must fix",
    )


class ReviewOutcome(BaseModel):
    """200 body for the approve endpoint."""

    complaint_id: uuid.UUID
    resolution_id: uuid.UUID
    version: int
    action: Literal["approved"]
    complaint_status: str
