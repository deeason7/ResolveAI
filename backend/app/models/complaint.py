"""Complaint ORM model — core entity of the system."""

import uuid
from datetime import datetime
from enum import Enum

from sqlmodel import Field, SQLModel

from app.models.types import EnumString


class ComplaintStatus(str, Enum):
    """Lifecycle of a complaint. Stored as VARCHAR(50), so adding a value here
    is a code change, not a migration.

    pending -> classified                       (low priority: stops here)
            -> escalated -> agent_triggered -> draft_ready  (guardrails passed)
                                            -> needs_review (agent failed/down)
    draft_ready -> resolved (human approves) or agent_triggered (human rejects)
    """

    pending = "pending"
    classified = "classified"
    escalated = "escalated"  # high priority: queued for the resolution agent
    agent_triggered = "agent_triggered"  # resolution agent is working on it
    draft_ready = "draft_ready"  # guardrail-passed draft awaiting human review
    needs_review = "needs_review"  # agent failed or unavailable; human takes over
    resolved = "resolved"  # human approved a resolution; case closed


class Complaint(SQLModel, table=True):
    __tablename__ = "complaints"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    # Source data from CFPB
    cfpb_complaint_id: str | None = Field(
        default=None,
        index=True,
        max_length=50,
        unique=True,
    )
    narrative: str
    product: str | None = Field(default=None, max_length=255)
    sub_product: str | None = Field(default=None, max_length=255)
    issue: str | None = Field(default=None, max_length=255)
    sub_issue: str | None = Field(default=None, max_length=255)
    company: str | None = Field(default=None, max_length=255)
    company_response: str | None = Field(default=None, max_length=255)
    state: str | None = Field(default=None, max_length=2)
    date_received: datetime | None = None

    # Classification outputs (filled by the SLM worker)
    # EnumString stores the enum as VARCHAR (matches the initial migration, no
    # native Postgres enum / migration overhead) and round-trips it back as the
    # enum, so reads carry .value and need no isinstance() guards.
    status: ComplaintStatus = Field(
        default=ComplaintStatus.pending,
        index=True,
        sa_type=EnumString(ComplaintStatus),
    )
    sentiment: str | None = Field(default=None, max_length=50)
    intent: str | None = Field(default=None, max_length=100)
    urgency: int | None = Field(default=None, ge=1, le=5)
    priority_score: float | None = Field(default=None, ge=0.0, le=1.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
