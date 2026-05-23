"""Complaint ORM model — core entity of the system."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

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
    cfpb_complaint_id: Optional[str] = Field(default=None, index=True, max_length=50)
    narrative: str
    product: Optional[str] = Field(default=None, max_length=255)
    sub_product: Optional[str] = Field(default=None, max_length=255)
    issue: Optional[str] = Field(default=None, max_length=255)
    sub_issue: Optional[str] = Field(default=None, max_length=255)
    company: Optional[str] = Field(default=None, max_length=255)
    company_response: Optional[str] = Field(default=None, max_length=255)
    state: Optional[str] = Field(default=None, max_length=2)
    date_received: Optional[datetime] = None

    # Classification outputs (filled by the SLM worker)
    status: ComplaintStatus = Field(default=ComplaintStatus.pending, index=True)
    sentiment: Optional[str] = Field(default=None, max_length=50)
    intent: Optional[str] = Field(default=None, max_length=100)
    urgency: Optional[int] = Field(default=None, ge=1, le=5)
    priority_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
