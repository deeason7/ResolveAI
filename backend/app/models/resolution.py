"""Resolution ORM model — each row is one drafted response for a complaint."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class GuardrailStatus(str, Enum):
    pending = "pending"
    passed = "passed"
    failed = "failed"
    escalated = "escalated"


class Resolution(SQLModel, table=True):
    __tablename__ = "resolutions"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    complaint_id: uuid.UUID = Field(foreign_key="complaints.id", index=True)

    version: int = Field(default=1)          # increments on re-generation
    draft_text: str
    guardrail_status: GuardrailStatus = Field(default=GuardrailStatus.pending)
    guardrail_notes: Optional[str] = None    # human-readable failure reason

    # Agent reasoning trace (JSON string — kept short for the DB, full trace in LLMLog)
    reasoning_summary: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
