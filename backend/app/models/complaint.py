"""Complaint ORM model — core entity of the system."""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import String
from sqlmodel import Field, SQLModel


class ComplaintStatus(str, Enum):
    pending = "pending"
    classified = "classified"
    resolved = "resolved"
    escalated = "escalated"


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
    # sa_type=String(50) keeps the Python enum for type safety but stores as
    # VARCHAR — matches the initial migration, avoids a native Postgres enum
    # type and the migration overhead that comes with it.
    status: ComplaintStatus = Field(
        default=ComplaintStatus.pending,
        index=True,
        sa_type=String(50),
    )
    sentiment: str | None = Field(default=None, max_length=50)
    intent: str | None = Field(default=None, max_length=100)
    urgency: int | None = Field(default=None, ge=1, le=5)
    priority_score: float | None = Field(default=None, ge=0.0, le=1.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
