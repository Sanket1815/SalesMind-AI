"""
SQLAlchemy ORM models for SalesMind AI.

Tables:
  leads          — raw inbound lead data
  pipeline_runs  — full pipeline result per lead (one-to-many with leads)
  guardian_logs  — individual Guardian safety check records
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Lead
# --------------------------------------------------------------------------- #

class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    company_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    contact_name: Mapped[str] = mapped_column(String(200), nullable=False)
    contact_email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    contact_title: Mapped[str] = mapped_column(String(200), nullable=False)
    company_website: Mapped[str | None] = mapped_column(String(500))
    industry: Mapped[str | None] = mapped_column(String(100), index=True)
    company_size_hint: Mapped[str | None] = mapped_column(String(100))
    initial_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    runs: Mapped[list["PipelineRun"]] = relationship(
        "PipelineRun", back_populates="lead", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Lead {self.company_name} / {self.contact_email}>"


# --------------------------------------------------------------------------- #
# PipelineRun
# --------------------------------------------------------------------------- #

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    lead_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("leads.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Research enrichment
    company_description: Mapped[str | None] = mapped_column(Text)
    employee_count: Mapped[str | None] = mapped_column(String(50))
    annual_revenue_estimate: Mapped[str | None] = mapped_column(String(100))
    funding_stage: Mapped[str | None] = mapped_column(String(50))
    company_location: Mapped[str | None] = mapped_column(String(200))
    tech_stack: Mapped[list | None] = mapped_column(JSONB)
    pain_points: Mapped[list | None] = mapped_column(JSONB)
    recent_news: Mapped[list | None] = mapped_column(JSONB)

    # Qualification
    qualification_score: Mapped[int | None] = mapped_column(Integer, index=True)
    is_qualified: Mapped[bool | None] = mapped_column(Boolean, index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    qualification_rationale: Mapped[str | None] = mapped_column(Text)
    bant_budget: Mapped[int | None] = mapped_column(Integer)
    bant_authority: Mapped[int | None] = mapped_column(Integer)
    bant_need: Mapped[int | None] = mapped_column(Integer)
    bant_timeline: Mapped[int | None] = mapped_column(Integer)
    model_used: Mapped[str | None] = mapped_column(String(100))

    # Outreach
    outreach_channel: Mapped[str | None] = mapped_column(String(20), index=True)
    outreach_subject: Mapped[str | None] = mapped_column(String(300))
    outreach_body: Mapped[str | None] = mapped_column(Text)
    outreach_tone: Mapped[str | None] = mapped_column(String(50))

    # Safety
    safety_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    human_review_required: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    safety_flags: Mapped[list] = mapped_column(JSONB, default=list)

    # Pipeline metadata
    current_step: Mapped[str] = mapped_column(String(50), default="unknown")
    pipeline_start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pipeline_end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    errors: Mapped[list] = mapped_column(JSONB, default=list)

    # Full state snapshot for debugging / replay
    full_state: Mapped[dict] = mapped_column(JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    lead: Mapped["Lead"] = relationship("Lead", back_populates="runs")
    guardian_logs: Mapped[list["GuardianLog"]] = relationship(
        "GuardianLog", back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<PipelineRun lead_id={self.lead_id} "
            f"score={self.qualification_score} qualified={self.is_qualified}>"
        )


# --------------------------------------------------------------------------- #
# GuardianLog
# --------------------------------------------------------------------------- #

class GuardianLog(Base):
    __tablename__ = "guardian_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    safe: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_type: Mapped[str | None] = mapped_column(String(100))
    reason: Mapped[str | None] = mapped_column(Text)
    raw_output: Mapped[str | None] = mapped_column(Text)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    run: Mapped["PipelineRun"] = relationship("PipelineRun", back_populates="guardian_logs")
