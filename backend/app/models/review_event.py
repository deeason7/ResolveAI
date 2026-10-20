"""One immutable decision about one saved response revision."""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlmodel import Field, SQLModel


class ReviewEvent(SQLModel, table=True):
    __tablename__ = "review_events"
    __table_args__ = (
        UniqueConstraint("resolution_id", name="uq_review_resolution"),
        UniqueConstraint("complaint_id", "case_version", name="uq_review_case_version"),
        CheckConstraint("action IN ('approved', 'rejected')", name="ck_review_action"),
        CheckConstraint(
            "case_version = expected_case_version + 1 AND expected_case_version >= 1",
            name="ck_review_case_version",
        ),
        CheckConstraint("input_revision >= 1", name="ck_review_input_revision"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    complaint_id: uuid.UUID = Field(foreign_key="complaints.id", index=True)
    resolution_id: uuid.UUID = Field(foreign_key="resolutions.id")
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True)
    action: str = Field(max_length=20)
    expected_case_version: int
    case_version: int
    input_revision: int
    draft_sha256: str = Field(max_length=64)
    feedback: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
