"""
Qualification Agent — scores a lead 0-100 using the BANT framework.

BANT: Budget (0-25) + Authority (0-25) + Need (0-25) + Timeline (0-25)

The agent also outputs:
- a confidence score (0.0-1.0) used by the dynamic model router
- a natural-language rationale for the score
- whether the lead is qualified (score >= settings.qualification_threshold)
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.graph.state import BANTScore, LeadState
from src.graph.router import select_model

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert B2B sales qualification analyst. Score a sales lead using the BANT framework.

BANT Scoring Guidelines (each component 0-25):

BUDGET (0-25):
- 25: Confirmed budget allocated, decision-maker authority confirmed
- 16-24: Budget exists, not yet formally allocated
- 6-15: Budget unclear, early-stage exploration
- 0-5: No budget signal or explicitly no budget

AUTHORITY (0-25):
- 25: C-level, VP, or confirmed decision-maker
- 16-24: Director/Manager with strong influence
- 6-15: Individual contributor with some influence
- 0-5: No authority signal

NEED (0-25):
- 25: Explicit pain point stated, actively evaluating solutions
- 16-24: Clear problem identified, solution consideration begun
- 6-15: Potential need inferred from context
- 0-5: No identifiable need

TIMELINE (0-25):
- 25: Immediate need (<30 days)
- 16-24: Short-term need (1-3 months)
- 6-15: Medium-term (3-12 months)
- 0-5: No timeline or >12 months

Return ONLY valid JSON:
{
  "bant_score": {
    "budget": <int 0-25>,
    "authority": <int 0-25>,
    "need": <int 0-25>,
    "timeline": <int 0-25>,
    "total": <int 0-100>
  },
  "confidence": <float 0.0-1.0, how confident you are in this scoring>,
  "rationale": "<concise 2-3 sentence explanation of the score>",
  "primary_strength": "<strongest qualification signal>",
  "primary_weakness": "<biggest qualification gap>"
}
"""


def _build_qualification_context(state: LeadState) -> str:
    parts = [
        f"Company: {state['company_name']}",
        f"Contact: {state['contact_name']} — {state['contact_title']}",
        f"Industry: {state.get('industry') or 'Unknown'}",
        f"Company Size: {state.get('company_size_hint') or state.get('employee_count') or 'Unknown'}",
    ]
    if state.get("company_description"):
        parts.append(f"Company Description: {state['company_description']}")
    if state.get("funding_stage"):
        parts.append(f"Funding Stage: {state['funding_stage']}")
    if state.get("annual_revenue_estimate"):
        parts.append(f"Revenue Estimate: {state['annual_revenue_estimate']}")
    if state.get("pain_points"):
        parts.append(f"Identified Pain Points: {', '.join(state['pain_points'])}")
    if state.get("tech_stack"):
        parts.append(f"Tech Stack: {', '.join(state['tech_stack'])}")
    if state.get("initial_message"):
        parts.append(f"\nLead's Initial Message:\n{state['initial_message']}")
    if state.get("recent_news"):
        news = "\n".join(f"• {n}" for n in state["recent_news"][:3])
        parts.append(f"\nRecent Company News:\n{news}")
    return "\n".join(parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _score_with_llm(context: str, model_id: str) -> dict[str, Any]:
    from src.models.watsonx_client import get_chat_model
    llm = get_chat_model(model_id=model_id)

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Score this lead:\n\n{context}"),
    ])
    raw = response.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    data = json.loads(raw)

    # Ensure total is sum of components
    bs = data["bant_score"]
    bs["total"] = bs["budget"] + bs["authority"] + bs["need"] + bs["timeline"]
    return data


def run_qualification_agent(state: LeadState) -> dict[str, Any]:
    """LangGraph node: score the lead and set qualification status."""
    logger.info(
        "Qualification agent started",
        extra={"lead_id": state["lead_id"], "company": state["company_name"]},
    )

    context = _build_qualification_context(state)

    # --- Dynamic model routing: first pass with fast model ---
    fast_model_id = settings.model_fast
    try:
        result = _score_with_llm(context, fast_model_id)
        confidence = float(result.get("confidence", 0.5))

        # If confidence is low, escalate to deep model for a better answer
        chosen_model = fast_model_id
        if confidence < settings.confidence_threshold:
            logger.info(
                "Low confidence, escalating to deep model",
                extra={"lead_id": state["lead_id"], "confidence": confidence},
            )
            chosen_model = settings.model_deep
            result = _score_with_llm(context, chosen_model)
            confidence = float(result.get("confidence", confidence))

        bant = result["bant_score"]
        score: int = bant["total"]
        is_qualified = score >= settings.qualification_threshold

        bant_typed: BANTScore = {
            "budget": bant["budget"],
            "authority": bant["authority"],
            "need": bant["need"],
            "timeline": bant["timeline"],
            "total": bant["total"],
        }

        logger.info(
            "Qualification complete",
            extra={
                "lead_id": state["lead_id"],
                "score": score,
                "qualified": is_qualified,
                "model": chosen_model,
            },
        )

        return {
            "current_step": "qualification_complete",
            "bant_score": bant_typed,
            "qualification_score": score,
            "qualification_rationale": result.get("rationale", ""),
            "is_qualified": is_qualified,
            "confidence": confidence,
            "model_used_for_qualification": chosen_model,
        }

    except Exception as exc:
        logger.error(
            "Qualification agent failed",
            extra={"lead_id": state["lead_id"], "error": str(exc)},
        )
        return {
            "current_step": "qualification_failed",
            "errors": [f"Qualification agent error: {exc}"],
            "bant_score": None,
            "qualification_score": 0,
            "qualification_rationale": "Qualification failed due to an error.",
            "is_qualified": False,
            "confidence": 0.0,
            "model_used_for_qualification": fast_model_id,
        }
