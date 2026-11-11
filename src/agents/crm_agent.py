"""
CRM Agent — packages final pipeline state into a structured CRM record
and persists it to PostgreSQL.

Optionally also pushes to Salesforce / HubSpot (production hook in _push_to_external_crm).
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from src.graph.state import LeadState

logger = logging.getLogger(__name__)


def _build_crm_record(state: LeadState) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    bant = state.get("bant_score") or {}
    return {
        "lead_id": state["lead_id"],
        "created_at": now,
        "contact": {
            "name": state["contact_name"],
            "email": state["contact_email"],
            "title": state["contact_title"],
        },
        "company": {
            "name": state["company_name"],
            "website": state.get("company_website"),
            "industry": state.get("industry"),
            "description": state.get("company_description"),
            "employee_count": state.get("employee_count"),
            "annual_revenue_estimate": state.get("annual_revenue_estimate"),
            "funding_stage": state.get("funding_stage"),
            "location": state.get("company_location"),
            "tech_stack": state.get("tech_stack") or [],
        },
        "qualification": {
            "score": state.get("qualification_score"),
            "is_qualified": state.get("is_qualified", False),
            "confidence": state.get("confidence"),
            "rationale": state.get("qualification_rationale"),
            "bant": {
                "budget": bant.get("budget"),
                "authority": bant.get("authority"),
                "need": bant.get("need"),
                "timeline": bant.get("timeline"),
                "total": bant.get("total"),
            },
            "model_used": state.get("model_used_for_qualification"),
        },
        "outreach": {
            "channel": state.get("outreach_channel"),
            "subject": state.get("outreach_subject"),
            "body": state.get("outreach_body"),
            "tone": state.get("outreach_tone"),
        },
        "safety": {
            "passed": state.get("safety_passed", True),
            "flags": state.get("safety_flags") or [],
            "guardian_checks": len(state.get("guardian_checks") or []),
            "human_review_required": state.get("human_review_required", False),
        },
        "pipeline": {
            "start_time": state.get("pipeline_start_time"),
            "end_time": now,
            "errors": state.get("errors") or [],
            "final_step": state.get("current_step"),
        },
        "pain_points": state.get("pain_points") or [],
        "recent_news": state.get("recent_news") or [],
    }


async def _persist_to_db(state: LeadState, crm_record: dict[str, Any]) -> str | None:
    """Save lead + pipeline run to PostgreSQL. Returns run_id or None on failure."""
    from config.settings import settings
    if not settings.database_url:
        return None  # DB not configured — skip silently

    try:
        from src.db.database import get_async_session
        from src.db.repository import create_lead, get_lead, save_pipeline_run

        async with get_async_session() as session:
            # Upsert lead (create only if not already exists)
            existing = await get_lead(session, state["lead_id"])
            if not existing:
                await create_lead(session, dict(state))

            run = await save_pipeline_run(session, state["lead_id"], dict(state))
            return str(run.id)

    except Exception as exc:
        logger.warning(
            "DB persist failed (pipeline result still returned)",
            extra={"lead_id": state["lead_id"], "error": str(exc)},
        )
        return None


def _push_to_external_crm(record: dict[str, Any]) -> None:
    """Production hook — replace with Salesforce / HubSpot API call."""
    # salesforce_client.create_or_update_lead(record)
    # hubspot_client.upsert_contact(record)
    pass


def run_crm_agent(state: LeadState) -> dict[str, Any]:
    """LangGraph node: build CRM record and persist to PostgreSQL."""
    logger.info(
        "CRM agent started",
        extra={"lead_id": state["lead_id"], "company": state["company_name"]},
    )

    try:
        record = _build_crm_record(state)
        end_time = record["pipeline"]["end_time"]

        # Persist to PostgreSQL (best-effort — does not fail the pipeline)
        run_id = asyncio.run(_persist_to_db(state, record))

        _push_to_external_crm(record)

        logger.info(
            "CRM record saved",
            extra={
                "lead_id": state["lead_id"],
                "qualified": record["qualification"]["is_qualified"],
                "db_run_id": run_id,
            },
        )

        return {
            "current_step": "crm_complete",
            "crm_record": record,
            "crm_status": "created",
            "pipeline_end_time": end_time,
        }

    except Exception as exc:
        logger.error(
            "CRM agent failed",
            extra={"lead_id": state["lead_id"], "error": str(exc)},
        )
        return {
            "current_step": "crm_failed",
            "errors": [f"CRM agent error: {exc}"],
            "crm_record": None,
            "crm_status": "error",
        }
