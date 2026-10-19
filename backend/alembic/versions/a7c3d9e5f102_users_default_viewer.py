"""Default new database users to viewer without changing existing roles.

Revision ID: a7c3d9e5f102
Revises: b2ba15ec5d94
"""

import sqlalchemy as sa
from alembic import op

revision = "a7c3d9e5f102"
down_revision = "b2ba15ec5d94"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "users",
        "role",
        existing_type=sa.String(50),
        existing_nullable=False,
        server_default="viewer",
    )


def downgrade() -> None:
    # Restores the old insert default only; never changes an existing account.
    op.alter_column(
        "users",
        "role",
        existing_type=sa.String(50),
        existing_nullable=False,
        server_default="analyst",
    )
