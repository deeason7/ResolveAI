"""Bind reviews to immutable drafts and separate case/input versions.

Revision ID: e6b4a9d2c705
Revises: c4a8e2d9f601
"""

import sqlalchemy as sa
from alembic import op

revision = "e6b4a9d2c705"
down_revision = "c4a8e2d9f601"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for name in ("case_version", "input_revision"):
        op.add_column("complaints", sa.Column(name, sa.Integer(), nullable=False, server_default="1"))
        op.create_check_constraint(f"ck_complaint_{name}", "complaints", f"{name} >= 1")
    op.add_column("complaints", sa.Column("legacy_approval_outcome_unknown", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.execute("UPDATE complaints SET legacy_approval_outcome_unknown = true WHERE status = 'resolved'")
    op.add_column("resolutions", sa.Column("input_revision", sa.Integer(), nullable=False, server_default="1"))
    op.create_check_constraint("ck_resolution_input_revision", "resolutions", "input_revision >= 1")
    # A deterministic version baseline is not proof of historical draft inputs.
    op.add_column("resolutions", sa.Column("legacy_provenance", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.alter_column("resolutions", "legacy_provenance", server_default=sa.false())
    op.create_table("review_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("complaint_id", sa.Uuid(), sa.ForeignKey("complaints.id"), nullable=False),
        sa.Column("resolution_id", sa.Uuid(), sa.ForeignKey("resolutions.id"), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("expected_case_version", sa.Integer(), nullable=False),
        sa.Column("case_version", sa.Integer(), nullable=False),
        sa.Column("input_revision", sa.Integer(), nullable=False),
        sa.Column("draft_sha256", sa.String(64), nullable=False),
        sa.Column("feedback", sa.String()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("resolution_id", name="uq_review_resolution"),
        sa.UniqueConstraint("complaint_id", "case_version", name="uq_review_case_version"),
        sa.CheckConstraint("action IN ('approved', 'rejected')", name="ck_review_action"),
        sa.CheckConstraint("case_version = expected_case_version + 1 AND expected_case_version >= 1", name="ck_review_case_version"),
        sa.CheckConstraint("input_revision >= 1", name="ck_review_input_revision"),
    )
    op.create_index("ix_review_events_complaint_id", "review_events", ["complaint_id"])
    op.create_index("ix_review_events_user_id", "review_events", ["user_id"])
    op.execute("""
        CREATE FUNCTION refuse_review_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'review events are immutable'; END $$
    """)
    op.execute("CREATE TRIGGER review_events_immutable BEFORE UPDATE OR DELETE ON review_events FOR EACH ROW EXECUTE FUNCTION refuse_review_mutation()")
    op.execute("""
        CREATE FUNCTION protect_saved_draft() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'UPDATE' THEN
                RAISE EXCEPTION 'saved drafts are immutable; insert a new revision';
            END IF;
            IF EXISTS (SELECT 1 FROM review_events WHERE resolution_id = OLD.id) THEN
                RAISE EXCEPTION 'reviewed drafts cannot be deleted';
            END IF;
            RETURN OLD;
        END $$
    """)
    op.execute("CREATE TRIGGER resolutions_immutable BEFORE UPDATE OR DELETE ON resolutions FOR EACH ROW EXECUTE FUNCTION protect_saved_draft()")


def downgrade() -> None:
    if op.get_bind().scalar(sa.text("SELECT EXISTS (SELECT 1 FROM review_events)")):
        raise RuntimeError("Refusing to remove review history; use a compatible forward migration")
    op.execute("DROP TRIGGER resolutions_immutable ON resolutions")
    op.execute("DROP FUNCTION protect_saved_draft()")
    op.execute("DROP TRIGGER review_events_immutable ON review_events")
    op.execute("DROP FUNCTION refuse_review_mutation()")
    op.drop_table("review_events")
    op.drop_constraint("ck_resolution_input_revision", "resolutions")
    op.drop_column("resolutions", "legacy_provenance")
    op.drop_column("resolutions", "input_revision")
    op.drop_column("complaints", "legacy_approval_outcome_unknown")
    for name in ("input_revision", "case_version"):
        op.drop_constraint(f"ck_complaint_{name}", "complaints")
        op.drop_column("complaints", name)
