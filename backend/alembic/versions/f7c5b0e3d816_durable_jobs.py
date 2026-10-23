"""Persist jobs, attempts and leases without inventing legacy queue ownership.

Revision ID: f7c5b0e3d816
Revises: e6b4a9d2c705
"""

import sqlalchemy as sa
from alembic import op

revision = "f7c5b0e3d816"
down_revision = "e6b4a9d2c705"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("complaint_id", sa.Uuid(), sa.ForeignKey("complaints.id"), nullable=False),
        sa.Column("kind", sa.String(50), nullable=False),
        sa.Column("case_version", sa.Integer(), nullable=False),
        sa.Column("input_revision", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("request_fingerprint", sa.String(64)),
        sa.Column("requested_by", sa.Uuid(), sa.ForeignKey("users.id")),
        sa.Column("review_event_id", sa.Uuid(), sa.ForeignKey("review_events.id")),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("total_attempts", sa.Integer(), nullable=False),
        sa.Column("retry_generation", sa.Integer(), nullable=False),
        sa.Column("last_retry_attempts", sa.Integer()),
        sa.Column("next_attempt_at", sa.DateTime(), nullable=False),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_expires_at", sa.DateTime()),
        sa.Column("heartbeat_at", sa.DateTime()),
        sa.Column("last_error_code", sa.String(64)),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime()),
        sa.UniqueConstraint("idempotency_key", name="uq_job_idempotency_key"),
        sa.UniqueConstraint("kind", "complaint_id", "case_version", "input_revision", name="uq_job_intent"),
        sa.CheckConstraint("kind IN ('classification', 'resolution', 'index')", name="ck_job_kind"),
        sa.CheckConstraint("status IN ('queued', 'running', 'succeeded', 'failed', 'superseded')", name="ck_job_status"),
        sa.CheckConstraint("case_version >= 1 AND input_revision >= 1", name="ck_job_versions"),
        sa.CheckConstraint("attempts >= 0 AND max_attempts >= 1 AND attempts <= max_attempts AND total_attempts >= attempts AND retry_generation >= 0", name="ck_job_attempts"),
        sa.CheckConstraint("(status = 'running' AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL AND heartbeat_at IS NOT NULL) OR (status <> 'running' AND lease_token IS NULL AND lease_expires_at IS NULL)", name="ck_job_lease"),
    )
    op.create_index("ix_jobs_complaint_id", "jobs", ["complaint_id"])
    op.create_index("ix_jobs_claim", "jobs", ["kind", "status", "next_attempt_at", "created_at"])


def downgrade() -> None:
    if op.get_bind().scalar(sa.text("SELECT EXISTS (SELECT 1 FROM jobs)")):
        raise RuntimeError("Refusing to remove durable job history; use a compatible forward migration")
    op.drop_table("jobs")
