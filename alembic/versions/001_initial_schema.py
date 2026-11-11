"""Initial schema — leads, pipeline_runs, guardian_logs

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # leads
    # ------------------------------------------------------------------ #
    op.create_table(
        "leads",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("company_name", sa.String(200), nullable=False),
        sa.Column("contact_name", sa.String(200), nullable=False),
        sa.Column("contact_email", sa.String(320), nullable=False),
        sa.Column("contact_title", sa.String(200), nullable=False),
        sa.Column("company_website", sa.String(500), nullable=True),
        sa.Column("industry", sa.String(100), nullable=True),
        sa.Column("company_size_hint", sa.String(100), nullable=True),
        sa.Column("initial_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_leads_company_name", "leads", ["company_name"])
    op.create_index("ix_leads_contact_email", "leads", ["contact_email"])
    op.create_index("ix_leads_industry", "leads", ["industry"])
    op.create_index("ix_leads_created_at", "leads", ["created_at"])

    # ------------------------------------------------------------------ #
    # pipeline_runs
    # ------------------------------------------------------------------ #
    op.create_table(
        "pipeline_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "lead_id",
            UUID(as_uuid=True),
            sa.ForeignKey("leads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Research
        sa.Column("company_description", sa.Text, nullable=True),
        sa.Column("employee_count", sa.String(50), nullable=True),
        sa.Column("annual_revenue_estimate", sa.String(100), nullable=True),
        sa.Column("funding_stage", sa.String(50), nullable=True),
        sa.Column("company_location", sa.String(200), nullable=True),
        sa.Column("tech_stack", JSONB, nullable=True),
        sa.Column("pain_points", JSONB, nullable=True),
        sa.Column("recent_news", JSONB, nullable=True),
        # Qualification
        sa.Column("qualification_score", sa.Integer, nullable=True),
        sa.Column("is_qualified", sa.Boolean, nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("qualification_rationale", sa.Text, nullable=True),
        sa.Column("bant_budget", sa.Integer, nullable=True),
        sa.Column("bant_authority", sa.Integer, nullable=True),
        sa.Column("bant_need", sa.Integer, nullable=True),
        sa.Column("bant_timeline", sa.Integer, nullable=True),
        sa.Column("model_used", sa.String(100), nullable=True),
        # Outreach
        sa.Column("outreach_channel", sa.String(20), nullable=True),
        sa.Column("outreach_subject", sa.String(300), nullable=True),
        sa.Column("outreach_body", sa.Text, nullable=True),
        sa.Column("outreach_tone", sa.String(50), nullable=True),
        # Safety
        sa.Column("safety_passed", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("human_review_required", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("safety_flags", JSONB, nullable=False, server_default="[]"),
        # Metadata
        sa.Column("current_step", sa.String(50), nullable=False, server_default="unknown"),
        sa.Column("pipeline_start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pipeline_end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("errors", JSONB, nullable=False, server_default="[]"),
        sa.Column("full_state", JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_pipeline_runs_lead_id", "pipeline_runs", ["lead_id"])
    op.create_index("ix_pipeline_runs_is_qualified", "pipeline_runs", ["is_qualified"])
    op.create_index("ix_pipeline_runs_human_review", "pipeline_runs", ["human_review_required"])
    op.create_index("ix_pipeline_runs_outreach_channel", "pipeline_runs", ["outreach_channel"])
    op.create_index("ix_pipeline_runs_created_at", "pipeline_runs", ["created_at"])

    # ------------------------------------------------------------------ #
    # guardian_logs
    # ------------------------------------------------------------------ #
    op.create_table(
        "guardian_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(50), nullable=False),
        sa.Column("safe", sa.Boolean, nullable=False),
        sa.Column("risk_type", sa.String(100), nullable=True),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("raw_output", sa.Text, nullable=True),
        sa.Column(
            "checked_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_guardian_logs_run_id", "guardian_logs", ["run_id"])


def downgrade() -> None:
    op.drop_table("guardian_logs")
    op.drop_table("pipeline_runs")
    op.drop_table("leads")
