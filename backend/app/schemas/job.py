"""Safe operational job responses omit request payloads and lease credentials."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.job import JobKind, JobStatus


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    complaint_id: uuid.UUID
    kind: JobKind
    case_version: int
    input_revision: int
    status: JobStatus
    attempts: int
    max_attempts: int
    total_attempts: int
    retry_generation: int
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime
    next_attempt_at: datetime
    heartbeat_at: datetime | None
    completed_at: datetime | None


class JobList(BaseModel):
    items: list[JobRead]
    total: int
    limit: int
    offset: int


class RetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_attempts: int = Field(ge=1, strict=True)
    expected_retry_generation: int = Field(ge=0, strict=True)


class JobQueueInfo(BaseModel):
    kind: JobKind
    queued: int = 0
    running: int = 0
    failed: int = 0
    succeeded: int = 0
    superseded: int = 0
    oldest_pending_age_seconds: float | None = None
