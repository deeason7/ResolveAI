"""Durable pipeline counts and job health."""

from pydantic import BaseModel

from app.models.job import JobKind
from app.schemas.job import JobQueueInfo


class WorkspaceBoard(BaseModel):
    pending: int
    classified: int
    escalated: int
    agent_triggered: int
    draft_ready: int
    needs_review: int
    response_approved: int
    resolved: int
    total: int
    classification_unavailable: int
    jobs: list[JobQueueInfo]
    admission_limit: int


class EnqueueResult(BaseModel):
    enqueued: int
    kind: JobKind
