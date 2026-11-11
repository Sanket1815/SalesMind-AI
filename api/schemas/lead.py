"""
Pydantic schemas for the SalesMind AI REST API.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


# --------------------------------------------------------------------------- #
# Request schemas
# --------------------------------------------------------------------------- #

class LeadRequest(BaseModel):
    """Inbound lead submitted to the pipeline."""

    company_name: str = Field(..., min_length=1, max_length=200)
    contact_name: str = Field(..., min_length=1, max_length=200)
    contact_email: EmailStr
    contact_title: str = Field(..., min_length=1, max_length=200)
    company_website: Optional[str] = Field(None, max_length=500)
    industry: Optional[str] = Field(None, max_length=100)
    company_size_hint: Optional[str] = Field(None, max_length=100)
    initial_message: Optional[str] = Field(None, max_length=5000)

    @field_validator("company_website")
    @classmethod
    def validate_website(cls, v: Optional[str]) -> Optional[str]:
        if v and not (v.startswith("http://") or v.startswith("https://")):
            return f"https://{v}"
        return v


class BatchLeadRequest(BaseModel):
    """Batch submission of multiple leads."""
    leads: list[LeadRequest] = Field(..., min_length=1, max_length=50)


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #

class BANTScoreResponse(BaseModel):
    budget: int
    authority: int
    need: int
    timeline: int
    total: int


class QualificationResponse(BaseModel):
    score: int
    is_qualified: bool
    confidence: float
    rationale: Optional[str]
    bant: Optional[BANTScoreResponse]
    model_used: Optional[str]


class OutreachResponse(BaseModel):
    channel: Optional[str]
    subject: Optional[str]
    body: Optional[str]
    tone: Optional[str]


class SafetyResponse(BaseModel):
    passed: bool
    flags: list[str]
    guardian_checks_count: int
    human_review_required: bool


class PipelineResponse(BaseModel):
    """Full pipeline result returned to the caller."""
    lead_id: str
    company_name: str
    contact_name: str
    contact_email: str

    qualification: QualificationResponse
    outreach: OutreachResponse
    safety: SafetyResponse

    pipeline_start_time: str
    pipeline_end_time: Optional[str]
    current_step: str
    errors: list[str]

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "PipelineResponse":
        bant_raw = state.get("bant_score")
        bant = BANTScoreResponse(**bant_raw) if bant_raw else None

        return cls(
            lead_id=state["lead_id"],
            company_name=state["company_name"],
            contact_name=state["contact_name"],
            contact_email=state["contact_email"],
            qualification=QualificationResponse(
                score=state.get("qualification_score") or 0,
                is_qualified=state.get("is_qualified") or False,
                confidence=state.get("confidence") or 0.0,
                rationale=state.get("qualification_rationale"),
                bant=bant,
                model_used=state.get("model_used_for_qualification"),
            ),
            outreach=OutreachResponse(
                channel=state.get("outreach_channel"),
                subject=state.get("outreach_subject"),
                body=state.get("outreach_body"),
                tone=state.get("outreach_tone"),
            ),
            safety=SafetyResponse(
                passed=state.get("safety_passed", True),
                flags=state.get("safety_flags") or [],
                guardian_checks_count=len(state.get("guardian_checks") or []),
                human_review_required=state.get("human_review_required", False),
            ),
            pipeline_start_time=state.get("pipeline_start_time", ""),
            pipeline_end_time=state.get("pipeline_end_time"),
            current_step=state.get("current_step", ""),
            errors=state.get("errors") or [],
        )


class BatchPipelineResponse(BaseModel):
    results: list[PipelineResponse]
    total: int
    qualified: int
    not_qualified: int
    errors: int


class HealthResponse(BaseModel):
    status: str
    version: str
    watsonx_configured: bool
