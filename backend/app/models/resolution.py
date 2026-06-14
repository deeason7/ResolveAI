"""Resolution ORM model — each row is one drafted response for a complaint."""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class GuardrailStatus(str, Enum):
    pending = "pending"
    passed = "passed"
    failed = "failed"
    escalated = "escalated"  # no verdict possible (e.g. LLM down) — human takes over


class Resolution(SQLModel, table=True):
    __tablename__ = "resolutions"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    complaint_id: uuid.UUID = Field(foreign_key="complaints.id", index=True)

    version: int = Field(default=1)  # increments on re-generation
    draft_text: str
    # sa_type=String(50): same enum-as-VARCHAR mapping as Complaint.status and
    # User.role — without it SQLModel emits a native Postgres enum cast that
    # doesn't match the VARCHAR column the initial migration created.
    guardrail_status: GuardrailStatus = Field(
        default=GuardrailStatus.pending,
        sa_type=String(50),
    )
    guardrail_notes: str | None = None  # human-readable failure reason
    # Structured violations (schemas.guardrails.GuardrailViolation dumps), so the
    # dashboard can filter by layer/code without parsing prose. JSONB on Postgres
    # for indexable filtering; plain JSON on SQLite in tests.
    guardrail_violations: list[dict] | None = Field(
        default=None,
        sa_type=JSON().with_variant(JSONB(), "postgresql"),
    )

    # Agent reasoning trace: newline-joined chain-of-thought steps (not JSON),
    # kept short for the DB; the full per-call trace lives in LLMLog.
    reasoning_summary: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
