"""Durable work owned by PostgreSQL, with a distinct identity for each intent."""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, CheckConstraint, Column, Index, UniqueConstraint
from sqlmodel import Field, SQLModel

from app.models.types import EnumString


class JobKind(str, Enum):
    classification = "classification"
    resolution = "resolution"
    index = "index"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    superseded = "superseded"


class Job(SQLModel, table=True):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_job_idempotency_key"),
        UniqueConstraint(
            "kind", "complaint_id", "case_version", "input_revision", name="uq_job_intent"
        ),
        CheckConstraint("kind IN ('classification', 'resolution', 'index')", name="ck_job_kind"),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'superseded')",
            name="ck_job_status",
        ),
        CheckConstraint("case_version >= 1 AND input_revision >= 1", name="ck_job_versions"),
        CheckConstraint(
            "attempts >= 0 AND max_attempts >= 1 AND attempts <= max_attempts "
            "AND total_attempts >= attempts AND retry_generation >= 0",
            name="ck_job_attempts",
        ),
        CheckConstraint(
            "(status = 'running' AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL "
            "AND heartbeat_at IS NOT NULL) OR (status <> 'running' AND lease_token IS NULL "
            "AND lease_expires_at IS NULL)",
            name="ck_job_lease",
        ),
        Index("ix_jobs_claim", "kind", "status", "next_attempt_at", "created_at"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    complaint_id: uuid.UUID = Field(foreign_key="complaints.id", index=True)
    kind: JobKind = Field(sa_type=EnumString(JobKind))
    case_version: int
    input_revision: int
    idempotency_key: str = Field(max_length=255)
    request_fingerprint: str | None = Field(default=None, max_length=64)
    requested_by: uuid.UUID | None = Field(default=None, foreign_key="users.id")
    review_event_id: uuid.UUID | None = Field(default=None, foreign_key="review_events.id")
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status: JobStatus = Field(default=JobStatus.queued, sa_type=EnumString(JobStatus))
    attempts: int = 0
    max_attempts: int = 3
    total_attempts: int = 0
    retry_generation: int = 0
    last_retry_attempts: int | None = None
    next_attempt_at: datetime = Field(default_factory=datetime.utcnow)
    lease_token: uuid.UUID | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    last_error_code: str | None = Field(default=None, max_length=64)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
