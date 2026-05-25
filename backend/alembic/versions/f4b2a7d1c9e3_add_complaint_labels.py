"""add complaint_labels table

Revision ID: f4b2a7d1c9e3
Revises: c8a4d1f5e3b7
Create Date: 2026-05-24

Provenance-tracked label store for the fine-tuning pipeline. Separate
from complaints.sentiment/intent/urgency (which the production
classifier writes to) so we can keep multiple labelers' outputs side by
side and re-label without destroying history. CASCADE on delete: a
label has no meaning without its complaint.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f4b2a7d1c9e3"
down_revision: Union[str, Sequence[str], None] = "c8a4d1f5e3b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "complaint_labels",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("complaint_id", sa.Uuid(), nullable=False),
        sa.Column("label_source", sa.String(length=100), nullable=False),
        sa.Column("sentiment", sa.String(length=20), nullable=False),
        sa.Column("intent", sa.String(length=40), nullable=False),
        sa.Column("urgency", sa.SmallInteger(), nullable=False),
        sa.Column("key_entities", sa.JSON(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("labeled_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["complaint_id"],
            ["complaints.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "complaint_id",
            "label_source",
            name="uq_complaint_labels_complaint_source",
        ),
    )
    op.create_index(
        "ix_complaint_labels_complaint_id",
        "complaint_labels",
        ["complaint_id"],
    )
    op.create_index(
        "ix_complaint_labels_label_source",
        "complaint_labels",
        ["label_source"],
    )


def downgrade() -> None:
    op.drop_index("ix_complaint_labels_label_source", table_name="complaint_labels")
    op.drop_index("ix_complaint_labels_complaint_id", table_name="complaint_labels")
    op.drop_table("complaint_labels")
