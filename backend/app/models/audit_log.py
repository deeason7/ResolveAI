"""AuditLog ORM model — immutable record of security-relevant user actions."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_logs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id", index=True)

    event: str = Field(max_length=100, index=True)   # e.g. "login", "view_complaint", "export"
    ip_hash: Optional[str] = Field(default=None, max_length=64)   # SHA-256 of raw IP
    metadata_: Optional[dict] = Field(default=None, sa_column=Column("metadata", JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow)
