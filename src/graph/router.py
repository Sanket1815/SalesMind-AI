"""
Dynamic model router for the SalesMind AI pipeline.

Routing logic:
  - If estimated deal size is large (enterprise signals) → always use deep model
  - If previous confidence was low → escalate to deep model
  - Otherwise → use fast model (lower latency + cost)

This module also provides the conditional edge functions used by LangGraph
to decide which node to go to after a guardian check.
"""

from __future__ import annotations

import logging
from typing import Literal

from config.settings import settings
from src.graph.state import LeadState

logger = logging.getLogger(__name__)

# Keywords that signal enterprise/large deal → route to deep model
_ENTERPRISE_SIGNALS = {
    "enterprise",
    "fortune 500",
    "fortune500",
    "global",
    "multinational",
    "ipo",
    "series d",
    "series e",
    "public company",
    "nyse",
    "nasdaq",
}

_EXEC_TITLES = {"ceo", "cto", "coo", "cfo", "president", "founder", "co-founder"}


def _has_enterprise_signal(state: LeadState) -> bool:
    text_corpus = " ".join(filter(None, [
        state.get("company_description") or "",
        state.get("funding_stage") or "",
        state.get("employee_count") or "",
        state.get("annual_revenue_estimate") or "",
        state.get("company_size_hint") or "",
    ])).lower()
    return any(sig in text_corpus for sig in _ENTERPRISE_SIGNALS)


def select_model(state: LeadState) -> str:
    """
    Choose which model ID to use for the qualification step.
    Called by the qualification agent before scoring.
    """
    if _has_enterprise_signal(state):
        logger.info(
            "Enterprise signal detected → deep model",
            extra={"lead_id": state["lead_id"]},
        )
        return settings.model_deep

    title = (state.get("contact_title") or "").lower()
    if any(kw in title for kw in _EXEC_TITLES):
        logger.info(
            "Executive contact → deep model",
            extra={"lead_id": state["lead_id"]},
        )
        return settings.model_deep

    return settings.model_fast


# --------------------------------------------------------------------------- #
# LangGraph conditional edge functions
# --------------------------------------------------------------------------- #

def route_after_qualification_guardian(
    state: LeadState,
) -> Literal["outreach", "human_review", "crm_update"]:
    """After the qualification guardian check, decide next node."""
    if state.get("human_review_required"):
        return "human_review"
    if state.get("is_qualified"):
        return "outreach"
    # Not qualified → skip outreach, go straight to CRM
    return "crm_update"


def route_after_outreach_guardian(
    state: LeadState,
) -> Literal["crm_update", "human_review"]:
    """After the outreach guardian check, decide next node."""
    if state.get("human_review_required"):
        return "human_review"
    return "crm_update"
