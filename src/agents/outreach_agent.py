"""
Outreach Agent — generates a personalised first-touch email or LinkedIn message.

Personalization draws on:
  - company description, pain points, recent news
  - contact title (determines tone and messaging angle)
  - qualification score (only runs for qualified leads)
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.graph.state import LeadState
from src.models.watsonx_client import get_deep_model

logger = logging.getLogger(__name__)

_SYSTEM_EMAIL = """You are an elite B2B sales copywriter. Write a cold outreach email that feels personal, not templated.

Rules:
- Subject line: compelling, <60 chars, no clickbait
- Body: 120-180 words max
- Open with a specific, relevant hook (news, pain point, or industry insight)
- One clear value proposition relevant to their role and pain points
- Soft CTA: suggest a 15-min call, NOT a hard sell
- Tone: professional but conversational — not robotic, not overly casual
- Never mention the sender's company features/pricing in the first touch
- Sign off naturally

Return JSON:
{
  "subject": "<email subject line>",
  "body": "<full email body with greeting and sign-off>",
  "channel": "email",
  "tone": "conversational"
}
"""

_SYSTEM_LINKEDIN = """You are an expert B2B LinkedIn outreach specialist. Write a connection request message.

Rules:
- Max 300 characters (LinkedIn connection request limit)
- Reference something specific about their company or role
- Clear but soft reason for connecting
- No links, no attachments mentioned
- Professional, not salesy

Return JSON:
{
  "subject": null,
  "body": "<LinkedIn message, max 300 chars>",
  "channel": "linkedin",
  "tone": "professional"
}
"""


def _pick_channel(contact_title: str) -> str:
    """C-suite and VPs get LinkedIn; others get email."""
    title_lower = (contact_title or "").lower()
    exec_keywords = {"ceo", "cto", "coo", "cfo", "founder", "president", "vp", "vice president"}
    if any(kw in title_lower for kw in exec_keywords):
        return "linkedin"
    return "email"


def _build_outreach_context(state: LeadState) -> str:
    parts = [
        f"Contact: {state['contact_name']}, {state['contact_title']} at {state['company_name']}",
        f"Industry: {state.get('industry') or 'Unknown'}",
    ]
    if state.get("company_description"):
        parts.append(f"About the company: {state['company_description']}")
    if state.get("pain_points"):
        parts.append(f"Pain points identified: {', '.join(state['pain_points'])}")
    if state.get("recent_news"):
        parts.append(f"Recent news: {state['recent_news'][0]}")
    if state.get("funding_stage"):
        parts.append(f"Funding stage: {state['funding_stage']}")
    if state.get("employee_count"):
        parts.append(f"Company size: {state['employee_count']} employees")
    if state.get("qualification_rationale"):
        parts.append(f"Qualification insight: {state['qualification_rationale']}")
    return "\n".join(parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _generate_outreach(context: str, channel: str) -> dict[str, Any]:
    import json

    llm = get_deep_model()
    system_prompt = _SYSTEM_EMAIL if channel == "email" else _SYSTEM_LINKEDIN
    user_msg = (
        f"Write a personalised {channel} outreach message for this lead:\n\n{context}\n\n"
        "Return ONLY valid JSON."
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ])
    raw = response.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


def run_outreach_agent(state: LeadState) -> dict[str, Any]:
    """LangGraph node: generate personalised outreach for qualified leads."""
    if not state.get("is_qualified"):
        logger.info(
            "Skipping outreach — lead not qualified",
            extra={"lead_id": state["lead_id"]},
        )
        return {
            "current_step": "outreach_skipped",
            "outreach_subject": None,
            "outreach_body": None,
            "outreach_channel": None,
            "outreach_tone": None,
        }

    logger.info(
        "Outreach agent started",
        extra={"lead_id": state["lead_id"], "company": state["company_name"]},
    )

    channel = _pick_channel(state.get("contact_title", ""))
    context = _build_outreach_context(state)

    try:
        result = _generate_outreach(context, channel)
        logger.info(
            "Outreach generated",
            extra={"lead_id": state["lead_id"], "channel": channel},
        )
        return {
            "current_step": "outreach_complete",
            "outreach_subject": result.get("subject"),
            "outreach_body": result.get("body"),
            "outreach_channel": result.get("channel", channel),
            "outreach_tone": result.get("tone"),
        }

    except Exception as exc:
        logger.error(
            "Outreach agent failed",
            extra={"lead_id": state["lead_id"], "error": str(exc)},
        )
        return {
            "current_step": "outreach_failed",
            "errors": [f"Outreach agent error: {exc}"],
            "outreach_subject": None,
            "outreach_body": None,
            "outreach_channel": None,
            "outreach_tone": None,
        }
