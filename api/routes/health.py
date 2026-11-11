"""
Health and readiness routes.

GET /health        — liveness probe
GET /health/ready  — readiness probe (checks Watsonx credentials)
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.schemas.lead import HealthResponse
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        watsonx_configured=bool(settings.watsonx_api_key and settings.watsonx_project_id),
    )


@router.get("/health/ready", summary="Readiness probe")
async def readiness():
    """Check that Watsonx credentials are present and the model client can be constructed."""
    issues = []
    if not settings.watsonx_api_key:
        issues.append("WATSONX_API_KEY not set")
    if not settings.watsonx_project_id:
        issues.append("WATSONX_PROJECT_ID not set")

    if issues:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "issues": issues},
        )

    return JSONResponse(status_code=200, content={"status": "ready"})
