"""unique resolution (complaint_id, version)

Revision ID: b2ba15ec5d94
Revises: d4f8c2b7e1a9
Create Date: 2026-06-14

Versions are assigned max(version)+1 per complaint — a read-modify-write that
two concurrent generations could race, both writing the same next version. This
composite UNIQUE makes Postgres reject the duplicate, so the invariant ("one row
per complaint+version") holds at the database, not merely by convention.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "b2ba15ec5d94"
down_revision: Union[str, Sequence[str], None] = "d4f8c2b7e1a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_resolution_complaint_version",
        "resolutions",
        ["complaint_id", "version"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_resolution_complaint_version",
        "resolutions",
        type_="unique",
    )
