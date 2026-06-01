"""AuditLog ORM model — immutable record of security-relevant user actions."""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_logs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID | None = Field(default=None, foreign_key="users.id", index=True)

    event: str = Field(max_length=100, index=True)  # e.g. "login", "view_complaint", "export"
    ip_hash: str | None = Field(default=None, max_length=64)  # SHA-256 of raw IP
    metadata_: dict | None = Field(default=None, sa_column=Column("metadata", JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow)
