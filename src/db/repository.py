"""
Repository layer — all database read/write operations.

Keeps SQL out of agents and API routes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models import GuardianLog, Lead, PipelineRun


# --------------------------------------------------------------------------- #
# Lead CRUD
# --------------------------------------------------------------------------- #

async def create_lead(session: AsyncSession, lead_data: dict[str, Any]) -> Lead:
    """Insert a new lead record. Returns the persisted Lead."""
    lead = Lead(
        id=uuid.UUID(lead_data["lead_id"]) if isinstance(lead_data.get("lead_id"), str) else uuid.uuid4(),
        company_name=lead_data["company_name"],
        contact_name=lead_data["contact_name"],
        contact_email=lead_data["contact_email"],
        contact_title=lead_data.get("contact_title", ""),
        company_website=lead_data.get("company_website"),
        industry=lead_data.get("industry"),
        company_size_hint=lead_data.get("company_size_hint"),
        initial_message=lead_data.get("initial_message"),
    )
    session.add(lead)
    await session.flush()
    return lead


async def get_lead(session: AsyncSession, lead_id: str) -> Lead | None:
    result = await session.execute(
        select(Lead)
        .where(Lead.id == uuid.UUID(lead_id))
        .options(selectinload(Lead.runs))
    )
    return result.scalar_one_or_none()


async def list_leads(
    session: AsyncSession,
    qualified: bool | None = None,
    industry: str | None = None,
    human_review: bool | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[Lead]:
    query = (
        select(Lead)
        .join(Lead.runs, isouter=True)
        .options(selectinload(Lead.runs))
        .order_by(desc(Lead.created_at))
        .limit(limit)
        .offset(offset)
    )
    if qualified is not None:
        query = query.where(PipelineRun.is_qualified == qualified)
    if industry:
        query = query.where(Lead.industry.ilike(f"%{industry}%"))
    if human_review is not None:
        query = query.where(PipelineRun.human_review_required == human_review)

    result = await session.execute(query)
    return list(result.scalars().unique())


# --------------------------------------------------------------------------- #
# PipelineRun CRUD
# --------------------------------------------------------------------------- #

async def save_pipeline_run(
    session: AsyncSession,
    lead_id: str,
    state: dict[str, Any],
) -> PipelineRun:
    """Persist a completed pipeline state as a PipelineRun record."""

    bant = state.get("bant_score") or {}

    def _parse_dt(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    run = PipelineRun(
        lead_id=uuid.UUID(lead_id),
        # Research
        company_description=state.get("company_description"),
        employee_count=state.get("employee_count"),
        annual_revenue_estimate=state.get("annual_revenue_estimate"),
        funding_stage=state.get("funding_stage"),
        company_location=state.get("company_location"),
        tech_stack=state.get("tech_stack") or [],
        pain_points=state.get("pain_points") or [],
        recent_news=state.get("recent_news") or [],
        # Qualification
        qualification_score=state.get("qualification_score"),
        is_qualified=state.get("is_qualified"),
        confidence=state.get("confidence"),
        qualification_rationale=state.get("qualification_rationale"),
        bant_budget=bant.get("budget"),
        bant_authority=bant.get("authority"),
        bant_need=bant.get("need"),
        bant_timeline=bant.get("timeline"),
        model_used=state.get("model_used_for_qualification"),
        # Outreach
        outreach_channel=state.get("outreach_channel"),
        outreach_subject=state.get("outreach_subject"),
        outreach_body=state.get("outreach_body"),
        outreach_tone=state.get("outreach_tone"),
        # Safety
        safety_passed=state.get("safety_passed", True),
        human_review_required=state.get("human_review_required", False),
        safety_flags=state.get("safety_flags") or [],
        # Metadata
        current_step=state.get("current_step", "unknown"),
        pipeline_start_time=_parse_dt(state.get("pipeline_start_time")),
        pipeline_end_time=_parse_dt(state.get("pipeline_end_time")),
        errors=state.get("errors") or [],
        full_state={k: v for k, v in state.items() if k != "guardian_checks"},
    )
    session.add(run)
    await session.flush()

    # Persist individual Guardian check records
    for check in state.get("guardian_checks") or []:
        log = GuardianLog(
            run_id=run.id,
            stage=check.get("stage", "unknown"),
            safe=check.get("safe", True),
            risk_type=check.get("risk_type"),
            reason=check.get("reason"),
            raw_output=check.get("raw_output"),
        )
        session.add(log)

    await session.flush()
    return run


async def get_pipeline_run(session: AsyncSession, run_id: str) -> PipelineRun | None:
    result = await session.execute(
        select(PipelineRun)
        .where(PipelineRun.id == uuid.UUID(run_id))
        .options(selectinload(PipelineRun.guardian_logs))
    )
    return result.scalar_one_or_none()


async def get_latest_run_for_lead(session: AsyncSession, lead_id: str) -> PipelineRun | None:
    result = await session.execute(
        select(PipelineRun)
        .where(PipelineRun.lead_id == uuid.UUID(lead_id))
        .order_by(desc(PipelineRun.created_at))
        .limit(1)
        .options(selectinload(PipelineRun.guardian_logs))
    )
    return result.scalar_one_or_none()


async def list_pipeline_runs(
    session: AsyncSession,
    qualified: bool | None = None,
    human_review: bool | None = None,
    outreach_channel: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[PipelineRun]:
    query = (
        select(PipelineRun)
        .order_by(desc(PipelineRun.created_at))
        .limit(limit)
        .offset(offset)
        .options(selectinload(PipelineRun.lead))
    )
    if qualified is not None:
        query = query.where(PipelineRun.is_qualified == qualified)
    if human_review is not None:
        query = query.where(PipelineRun.human_review_required == human_review)
    if outreach_channel:
        query = query.where(PipelineRun.outreach_channel == outreach_channel)

    result = await session.execute(query)
    return list(result.scalars().unique())


# --------------------------------------------------------------------------- #
# Stats helper
# --------------------------------------------------------------------------- #

async def get_pipeline_stats(session: AsyncSession) -> dict[str, Any]:
    from sqlalchemy import func

    total = await session.scalar(select(func.count(PipelineRun.id)))
    qualified = await session.scalar(
        select(func.count(PipelineRun.id)).where(PipelineRun.is_qualified.is_(True))
    )
    review = await session.scalar(
        select(func.count(PipelineRun.id)).where(PipelineRun.human_review_required.is_(True))
    )
    avg_score = await session.scalar(
        select(func.avg(PipelineRun.qualification_score)).where(
            PipelineRun.qualification_score.is_not(None)
        )
    )

    return {
        "total_runs": total or 0,
        "qualified": qualified or 0,
        "not_qualified": (total or 0) - (qualified or 0),
        "human_review_pending": review or 0,
        "avg_qualification_score": round(float(avg_score), 1) if avg_score else None,
        "qualification_rate": round(qualified / total, 3) if total else 0.0,
    }
