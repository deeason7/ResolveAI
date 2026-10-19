"""Track classification outcomes without inventing legacy provenance.

Revision ID: c4a8e2d9f601
Revises: a7c3d9e5f102
"""

import sqlalchemy as sa

from alembic import op

revision = "c4a8e2d9f601"
down_revision = "a7c3d9e5f102"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "complaints",
        sa.Column(
            "classification_outcome",
            sa.String(50),
            nullable=False,
            server_default="legacy_unknown",
        ),
    )
    # Only rows written by the new application carry pending provenance. All
    # existing rows retain unknown, even when their lifecycle says classified.
    op.alter_column("complaints", "classification_outcome", server_default="pending")
    op.create_index(
        "ix_complaints_classification_outcome", "complaints", ["classification_outcome"]
    )
    op.add_column("complaints", sa.Column("classification_source", sa.String(100)))
    op.add_column(
        "complaints",
        sa.Column("classification_stale", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("complaints", sa.Column("classification_attempted_at", sa.DateTime()))
    op.add_column("complaints", sa.Column("classified_at", sa.DateTime()))
    op.add_column("complaints", sa.Column("classification_error", sa.String(50)))
    op.add_column("complaints", sa.Column("classification_latency_ms", sa.Integer()))
    op.add_column("complaints", sa.Column("classification_was_fallback", sa.Boolean()))


def downgrade() -> None:
    # Destructive: callers must preserve a backup before dropping provenance.
    op.drop_index("ix_complaints_classification_outcome", table_name="complaints")
    for name in (
        "classification_was_fallback",
        "classification_latency_ms",
        "classification_error",
        "classified_at",
        "classification_attempted_at",
        "classification_stale",
        "classification_source",
        "classification_outcome",
    ):
        op.drop_column("complaints", name)
