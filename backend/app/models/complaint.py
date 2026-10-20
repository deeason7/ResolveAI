"""Complaint ORM model — core entity of the system."""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import CheckConstraint, false
from sqlmodel import Field, SQLModel

from app.models.types import EnumString


class ComplaintStatus(str, Enum):
    """Lifecycle of a complaint. Stored as VARCHAR(50), so adding a value here
    is a code change, not a migration.

    pending -> classified                       (low priority: stops here)
            -> escalated -> agent_triggered -> draft_ready  (guardrails passed)
                                            -> needs_review (agent failed/down)
    draft_ready -> response_approved (human approves) or agent_triggered (rejects)
    """

    pending = "pending"
    classified = "classified"
    escalated = "escalated"  # high priority: queued for the resolution agent
    agent_triggered = "agent_triggered"  # resolution agent is working on it
    draft_ready = "draft_ready"  # guardrail-passed draft awaiting human review
    needs_review = "needs_review"  # agent failed or unavailable; human takes over
    response_approved = "response_approved"  # approved text; delivery/outcome unrecorded
    resolved = "resolved"  # legacy approval; delivery and outcome unknown


class ClassificationState(str, Enum):
    """Latest classification outcome, independent of the case lifecycle."""

    pending = "pending"
    succeeded = "succeeded"
    unavailable = "unavailable"
    manual = "manual"
    legacy_unknown = "legacy_unknown"


class Complaint(SQLModel, table=True):
    __tablename__ = "complaints"
    __table_args__ = (
        CheckConstraint("case_version >= 1", name="ck_complaint_case_version"),
        CheckConstraint("input_revision >= 1", name="ck_complaint_input_revision"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    case_version: int = Field(default=1, sa_column_kwargs={"server_default": "1"})
    input_revision: int = Field(default=1, sa_column_kwargs={"server_default": "1"})
    legacy_approval_outcome_unknown: bool = Field(
        default=False, sa_column_kwargs={"server_default": false()}
    )

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

    classification_outcome: ClassificationState = Field(
        default=ClassificationState.pending,
        sa_type=EnumString(ClassificationState),
        sa_column_kwargs={"server_default": "pending"},
        index=True,
    )
    # Source describes the retained labels, not a failed replacement attempt.
    classification_source: str | None = Field(default=None, max_length=100)
    classification_stale: bool = Field(default=False, sa_column_kwargs={"server_default": false()})
    classification_attempted_at: datetime | None = None
    classified_at: datetime | None = None
    classification_error: str | None = Field(default=None, max_length=50)
    classification_latency_ms: int | None = None
    classification_was_fallback: bool | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
