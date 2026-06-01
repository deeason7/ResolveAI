"""initial schema

Revision ID: abad4bedd5a0
Revises:
Create Date: 2026-05-23

Creates all five core tables: users, complaints, resolutions, llm_logs, audit_logs.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "abad4bedd5a0"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False, server_default="analyst"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "complaints",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("cfpb_complaint_id", sa.String(length=50), nullable=True),
        sa.Column("narrative", sa.Text(), nullable=False),
        sa.Column("product", sa.String(length=255), nullable=True),
        sa.Column("sub_product", sa.String(length=255), nullable=True),
        sa.Column("issue", sa.String(length=255), nullable=True),
        sa.Column("sub_issue", sa.String(length=255), nullable=True),
        sa.Column("company", sa.String(length=255), nullable=True),
        sa.Column("company_response", sa.String(length=255), nullable=True),
        sa.Column("state", sa.String(length=2), nullable=True),
        sa.Column("date_received", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("sentiment", sa.String(length=50), nullable=True),
        sa.Column("intent", sa.String(length=100), nullable=True),
        sa.Column("urgency", sa.Integer(), nullable=True),
        sa.Column("priority_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_complaints_cfpb_complaint_id", "complaints", ["cfpb_complaint_id"])
    op.create_index("ix_complaints_status", "complaints", ["status"])

    op.create_table(
        "resolutions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("complaint_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("draft_text", sa.Text(), nullable=False),
        sa.Column("guardrail_status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("guardrail_notes", sa.Text(), nullable=True),
        sa.Column("reasoning_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["complaint_id"], ["complaints.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_resolutions_complaint_id", "resolutions", ["complaint_id"])

    op.create_table(
        "llm_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("complaint_id", sa.Uuid(), nullable=True),
        sa.Column("operation", sa.String(length=100), nullable=False),
        sa.Column("model_used", sa.String(length=100), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("was_fallback", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["complaint_id"], ["complaints.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_logs_complaint_id", "llm_logs", ["complaint_id"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("event", sa.String(length=100), nullable=False),
        sa.Column("ip_hash", sa.String(length=64), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_event", "audit_logs", ["event"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("llm_logs")
    op.drop_table("resolutions")
    op.drop_table("complaints")
    op.drop_table("users")
