"""
ComplaintLabel ORM model — teacher labels with provenance.

One row per (complaint, label_source). Persists the structured
classification produced by a labeler (LLM or human) together with the
operational metadata needed to reproduce, audit, or compare runs.
Separate from `complaints.sentiment/intent/urgency`, which are reserved
for the production classifier's outputs.
"""

import uuid
from datetime import datetime

from sqlalchemy import JSON, UniqueConstraint
from sqlmodel import Field, SQLModel


class ComplaintLabel(SQLModel, table=True):
    __tablename__ = "complaint_labels"
    __table_args__ = (
        UniqueConstraint(
            "complaint_id",
            "label_source",
            name="uq_complaint_labels_complaint_source",
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    complaint_id: uuid.UUID = Field(
        foreign_key="complaints.id",
        index=True,
    )

    # Provenance — identifies the labeler so we can re-label with a
    # different teacher later without overwriting the originals.
    label_source: str = Field(max_length=100, index=True)

    # The classification — string columns deliberately wider than current
    # vocab so adding a new sentiment/intent value is a code change, not
    # a migration. Validation lives at the schema layer (classification.py).
    sentiment: str = Field(max_length=20)
    intent: str = Field(max_length=40)
    urgency: int = Field(ge=1, le=5)
    key_entities: list[dict] = Field(default_factory=list, sa_type=JSON)
    reasoning: str

    # Operational metadata — populated when available, useful for cost
    # tracking and post-hoc analysis ("which complaints did the teacher
    # spend the most tokens on?"). Nullable because not every labeler
    # exposes these.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None

    labeled_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
