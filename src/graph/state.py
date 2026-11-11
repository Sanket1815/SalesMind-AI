"""
LangGraph state schema for the SalesMind AI pipeline.

All agent nodes read from and write into LeadState.
Reducers are annotated where list fields require append semantics.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


def _append(existing: list, new: list) -> list:
    """Reducer: append new items to existing list (no deduplication)."""
    return (existing or []) + (new or [])


class BANTScore(TypedDict):
    budget: int       # 0-25
    authority: int    # 0-25
    need: int         # 0-25
    timeline: int     # 0-25
    total: int        # 0-100


class GuardianCheck(TypedDict):
    stage: str        # e.g. "qualification", "outreach"
    safe: bool
    risk_type: Optional[str]
    reason: Optional[str]
    raw_output: str


class LeadState(TypedDict):
    # ------------------------------------------------------------------ #
    # Input fields (provided by caller)
    # ------------------------------------------------------------------ #
    lead_id: str
    company_name: str
    contact_name: str
    contact_email: str
    contact_title: str
    company_website: Optional[str]
    industry: Optional[str]
    company_size_hint: Optional[str]   # e.g. "50-200 employees"
    initial_message: Optional[str]     # any message the lead sent

    # ------------------------------------------------------------------ #
    # Research enrichment (populated by ResearchAgent)
    # ------------------------------------------------------------------ #
    company_description: Optional[str]
    employee_count: Optional[str]
    annual_revenue_estimate: Optional[str]
    funding_stage: Optional[str]
    tech_stack: Optional[list[str]]
    pain_points: Optional[list[str]]
    recent_news: Optional[list[str]]
    company_location: Optional[str]

    # ------------------------------------------------------------------ #
    # Qualification (populated by QualificationAgent)
    # ------------------------------------------------------------------ #
    bant_score: Optional[BANTScore]
    qualification_score: Optional[int]        # 0-100 total
    qualification_rationale: Optional[str]
    is_qualified: Optional[bool]
    confidence: Optional[float]               # model confidence 0.0-1.0
    model_used_for_qualification: Optional[str]

    # ------------------------------------------------------------------ #
    # Outreach (populated by OutreachAgent)
    # ------------------------------------------------------------------ #
    outreach_subject: Optional[str]
    outreach_body: Optional[str]
    outreach_channel: Optional[str]           # "email" | "linkedin"
    outreach_tone: Optional[str]              # "formal" | "conversational"

    # ------------------------------------------------------------------ #
    # CRM record (populated by CRMAgent)
    # ------------------------------------------------------------------ #
    crm_record: Optional[dict[str, Any]]
    crm_status: Optional[str]                 # "created" | "updated" | "skipped"

    # ------------------------------------------------------------------ #
    # Safety / Guardian
    # ------------------------------------------------------------------ #
    guardian_checks: Annotated[list[GuardianCheck], _append]
    safety_passed: bool
    human_review_required: bool
    safety_flags: Annotated[list[str], _append]

    # ------------------------------------------------------------------ #
    # Pipeline metadata
    # ------------------------------------------------------------------ #
    current_step: str
    pipeline_start_time: str
    pipeline_end_time: Optional[str]
    errors: Annotated[list[str], _append]
    retry_count: int


def initial_state(lead_data: dict[str, Any]) -> LeadState:
    """Build a fresh LeadState from raw lead input dict."""
    from datetime import datetime, timezone
    import uuid

    return LeadState(
        lead_id=lead_data.get("lead_id", str(uuid.uuid4())),
        company_name=lead_data["company_name"],
        contact_name=lead_data["contact_name"],
        contact_email=lead_data["contact_email"],
        contact_title=lead_data.get("contact_title", ""),
        company_website=lead_data.get("company_website"),
        industry=lead_data.get("industry"),
        company_size_hint=lead_data.get("company_size_hint"),
        initial_message=lead_data.get("initial_message"),
        # Research — empty
        company_description=None,
        employee_count=None,
        annual_revenue_estimate=None,
        funding_stage=None,
        tech_stack=None,
        pain_points=None,
        recent_news=None,
        company_location=None,
        # Qualification — empty
        bant_score=None,
        qualification_score=None,
        qualification_rationale=None,
        is_qualified=None,
        confidence=None,
        model_used_for_qualification=None,
        # Outreach — empty
        outreach_subject=None,
        outreach_body=None,
        outreach_channel=None,
        outreach_tone=None,
        # CRM — empty
        crm_record=None,
        crm_status=None,
        # Safety
        guardian_checks=[],
        safety_passed=True,
        human_review_required=False,
        safety_flags=[],
        # Metadata
        current_step="start",
        pipeline_start_time=datetime.now(timezone.utc).isoformat(),
        pipeline_end_time=None,
        errors=[],
        retry_count=0,
    )
