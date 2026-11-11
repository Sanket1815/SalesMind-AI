"""
Results API — read pipeline runs and lead records from PostgreSQL.

GET  /results/runs                      — paginated list of all pipeline runs
GET  /results/runs/{run_id}             — single pipeline run by ID
GET  /results/leads                     — paginated list of all leads
GET  /results/leads/{lead_id}           — single lead with its latest run
GET  /results/leads/{lead_id}/runs      — all runs for a lead
GET  /results/stats                     — aggregate pipeline statistics
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import get_async_session
from src.db.repository import (
    get_lead,
    get_latest_run_for_lead,
    get_pipeline_run,
    get_pipeline_stats,
    list_leads,
    list_pipeline_runs,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/results", tags=["results"])


# --------------------------------------------------------------------------- #
# Dependency — async DB session
# --------------------------------------------------------------------------- #

async def db_session():
    async with get_async_session() as session:
        yield session


# --------------------------------------------------------------------------- #
# Pipeline runs
# --------------------------------------------------------------------------- #

@router.get("/runs", summary="List all pipeline runs")
async def list_runs(
    qualified: Optional[bool] = Query(None, description="Filter by qualification status"),
    human_review: Optional[bool] = Query(None, description="Filter by human review flag"),
    channel: Optional[str] = Query(None, description="Filter by outreach channel (email|linkedin)"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(db_session),
):
    runs = await list_pipeline_runs(
        session,
        qualified=qualified,
        human_review=human_review,
        outreach_channel=channel,
        limit=limit,
        offset=offset,
    )
    return {
        "total": len(runs),
        "offset": offset,
        "limit": limit,
        "runs": [_serialize_run(r) for r in runs],
    }


@router.get("/runs/{run_id}", summary="Get a single pipeline run by ID")
async def get_run(run_id: str, session: AsyncSession = Depends(db_session)):
    run = await get_pipeline_run(session, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return _serialize_run(run, include_full_state=True)


# --------------------------------------------------------------------------- #
# Leads
# --------------------------------------------------------------------------- #

@router.get("/leads", summary="List all leads")
async def list_leads_endpoint(
    qualified: Optional[bool] = Query(None),
    industry: Optional[str] = Query(None),
    human_review: Optional[bool] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(db_session),
):
    leads = await list_leads(
        session,
        qualified=qualified,
        industry=industry,
        human_review=human_review,
        limit=limit,
        offset=offset,
    )
    return {
        "total": len(leads),
        "offset": offset,
        "limit": limit,
        "leads": [_serialize_lead(lead) for lead in leads],
    }


@router.get("/leads/{lead_id}", summary="Get a lead with its latest pipeline run")
async def get_lead_endpoint(lead_id: str, session: AsyncSession = Depends(db_session)):
    lead = await get_lead(session, lead_id)
    if not lead:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead not found")

    latest_run = await get_latest_run_for_lead(session, lead_id)
    return {
        "lead": _serialize_lead(lead),
        "latest_run": _serialize_run(latest_run) if latest_run else None,
    }


@router.get("/leads/{lead_id}/runs", summary="Get all pipeline runs for a lead")
async def get_lead_runs(lead_id: str, session: AsyncSession = Depends(db_session)):
    lead = await get_lead(session, lead_id)
    if not lead:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead not found")

    return {
        "lead_id": lead_id,
        "company_name": lead.company_name,
        "runs": [_serialize_run(r) for r in lead.runs],
    }


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

@router.get("/stats", summary="Aggregate pipeline statistics")
async def stats(session: AsyncSession = Depends(db_session)):
    return await get_pipeline_stats(session)


# --------------------------------------------------------------------------- #
# Serializers
# --------------------------------------------------------------------------- #

def _serialize_run(run, include_full_state: bool = False) -> dict:
    data = {
        "id": str(run.id),
        "lead_id": str(run.lead_id),
        "qualification": {
            "score": run.qualification_score,
            "is_qualified": run.is_qualified,
            "confidence": run.confidence,
            "rationale": run.qualification_rationale,
            "bant": {
                "budget": run.bant_budget,
                "authority": run.bant_authority,
                "need": run.bant_need,
                "timeline": run.bant_timeline,
                "total": (run.bant_budget or 0) + (run.bant_authority or 0)
                        + (run.bant_need or 0) + (run.bant_timeline or 0),
            },
            "model_used": run.model_used,
        },
        "outreach": {
            "channel": run.outreach_channel,
            "subject": run.outreach_subject,
            "body": run.outreach_body,
            "tone": run.outreach_tone,
        },
        "safety": {
            "passed": run.safety_passed,
            "flags": run.safety_flags or [],
            "human_review_required": run.human_review_required,
            "guardian_checks": [
                {
                    "stage": g.stage,
                    "safe": g.safe,
                    "risk_type": g.risk_type,
                    "reason": g.reason,
                }
                for g in (run.guardian_logs or [])
            ],
        },
        "research": {
            "company_description": run.company_description,
            "employee_count": run.employee_count,
            "funding_stage": run.funding_stage,
            "annual_revenue_estimate": run.annual_revenue_estimate,
            "tech_stack": run.tech_stack or [],
            "pain_points": run.pain_points or [],
            "recent_news": run.recent_news or [],
            "location": run.company_location,
        },
        "pipeline": {
            "current_step": run.current_step,
            "start_time": run.pipeline_start_time.isoformat() if run.pipeline_start_time else None,
            "end_time": run.pipeline_end_time.isoformat() if run.pipeline_end_time else None,
            "errors": run.errors or [],
        },
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }
    if include_full_state:
        data["full_state"] = run.full_state
    return data


def _serialize_lead(lead) -> dict:
    return {
        "id": str(lead.id),
        "company_name": lead.company_name,
        "contact_name": lead.contact_name,
        "contact_email": lead.contact_email,
        "contact_title": lead.contact_title,
        "company_website": lead.company_website,
        "industry": lead.industry,
        "company_size_hint": lead.company_size_hint,
        "initial_message": lead.initial_message,
        "created_at": lead.created_at.isoformat() if lead.created_at else None,
        "run_count": len(lead.runs) if lead.runs else 0,
    }
