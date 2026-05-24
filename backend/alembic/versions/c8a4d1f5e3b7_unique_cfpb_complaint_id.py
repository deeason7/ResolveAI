"""unique cfpb_complaint_id

Revision ID: c8a4d1f5e3b7
Revises: abad4bedd5a0
Create Date: 2026-05-23

Adds a UNIQUE constraint on complaints.cfpb_complaint_id so the bulk
import path can use ON CONFLICT DO NOTHING and stay idempotent across
re-runs. NULL values remain valid and may repeat (Postgres semantics).
"""

from typing import Sequence, Union

from alembic import op

revision: str = "c8a4d1f5e3b7"
down_revision: Union[str, Sequence[str], None] = "abad4bedd5a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_complaints_cfpb_complaint_id",
        "complaints",
        ["cfpb_complaint_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_complaints_cfpb_complaint_id",
        "complaints",
        type_="unique",
    )
