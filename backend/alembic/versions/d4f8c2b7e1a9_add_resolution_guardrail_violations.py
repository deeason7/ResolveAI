"""add guardrail_violations to resolutions

Revision ID: d4f8c2b7e1a9
Revises: f4b2a7d1c9e3
Create Date: 2026-06-10

Structured guardrail failures for each draft (list of layer/code/message
objects). Lives alongside guardrail_notes, which stays human-readable —
the dashboard's violation log filters on layer/code, and parsing that out
of prose would be backwards. JSONB so those filters can be indexed later.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d4f8c2b7e1a9"
down_revision: Union[str, Sequence[str], None] = "f4b2a7d1c9e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "resolutions",
        sa.Column(
            "guardrail_violations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("resolutions", "guardrail_violations")
