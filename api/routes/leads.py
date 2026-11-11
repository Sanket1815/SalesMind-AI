"""
Lead qualification API routes.

POST /leads/qualify          — run the full pipeline for one lead
POST /leads/qualify/batch    — run the pipeline for up to 50 leads (sequentially)
GET  /leads/{lead_id}/result — retrieve a previous result from the checkpoint store
"""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from api.schemas.lead import (
    BatchLeadRequest,
    BatchPipelineResponse,
    LeadRequest,
    PipelineResponse,
)
from src.graph.pipeline import arun_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/leads", tags=["leads"])


def _lead_request_to_dict(req: LeadRequest, lead_id: str) -> dict[str, Any]:
    return {
        "lead_id": lead_id,
        "company_name": req.company_name,
        "contact_name": req.contact_name,
        "contact_email": req.contact_email,
        "contact_title": req.contact_title,
        "company_website": req.company_website,
        "industry": req.industry,
        "company_size_hint": req.company_size_hint,
        "initial_message": req.initial_message,
    }


@router.post(
    "/qualify",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Qualify a single lead through the full AI pipeline",
)
async def qualify_lead(request: LeadRequest) -> PipelineResponse:
    lead_id = str(uuid.uuid4())
    lead_data = _lead_request_to_dict(request, lead_id)

    logger.info(
        "Qualify endpoint called",
        extra={"lead_id": lead_id, "company": request.company_name},
    )

    try:
        final_state = await arun_pipeline(lead_data, thread_id=lead_id)
        return PipelineResponse.from_state(final_state)
    except Exception as exc:
        logger.error("Pipeline error", extra={"lead_id": lead_id, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {exc}",
        )


@router.post(
    "/qualify/batch",
    response_model=BatchPipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Qualify multiple leads (up to 50) through the pipeline",
)
async def qualify_leads_batch(request: BatchLeadRequest) -> BatchPipelineResponse:
    logger.info("Batch qualify called", extra={"count": len(request.leads)})

    async def _run_one(lead_req: LeadRequest) -> PipelineResponse:
        lead_id = str(uuid.uuid4())
        lead_data = _lead_request_to_dict(lead_req, lead_id)
        try:
            final_state = await arun_pipeline(lead_data, thread_id=lead_id)
            return PipelineResponse.from_state(final_state)
        except Exception as exc:
            logger.error("Batch item failed", extra={"company": lead_req.company_name, "error": str(exc)})
            # Return a minimal error response for failed leads
            return PipelineResponse(
                lead_id=lead_id,
                company_name=lead_req.company_name,
                contact_name=lead_req.contact_name,
                contact_email=str(lead_req.contact_email),
                qualification=dict(
                    score=0, is_qualified=False, confidence=0.0,
                    rationale=None, bant=None, model_used=None
                ),
                outreach=dict(channel=None, subject=None, body=None, tone=None),
                safety=dict(passed=True, flags=[], guardian_checks_count=0, human_review_required=False),
                pipeline_start_time="",
                pipeline_end_time=None,
                current_step="error",
                errors=[str(exc)],
            )

    # Run concurrently (bounded by Watsonx rate limits in practice)
    tasks = [_run_one(lead) for lead in request.leads]
    results = await asyncio.gather(*tasks)

    qualified = sum(1 for r in results if r.qualification.is_qualified)
    errors = sum(1 for r in results if r.errors)

    return BatchPipelineResponse(
        results=list(results),
        total=len(results),
        qualified=qualified,
        not_qualified=len(results) - qualified,
        errors=errors,
    )
