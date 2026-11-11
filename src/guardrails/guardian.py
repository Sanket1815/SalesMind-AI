"""
Granite Guardian safety layer.

IBM Granite Guardian 3 is a specialised safety classifier.
It accepts a user/assistant conversation and outputs "safe" or "unsafe"
plus a risk label from its built-in taxonomy.

Risk types checked:
  - social_bias         : demographic bias or stereotyping
  - profanity           : offensive language
  - violence            : violent content
  - sexual_content      : explicit material
  - unethical_behavior  : deceptive or manipulative content
  - groundedness        : hallucinated / unverifiable claims
  - relevance           : off-topic responses

We wrap Guardian into a LangGraph node that:
1. Checks the most recent agent output (qualification rationale or outreach body)
2. Returns safe=True to continue, or flags for human review on repeated failures
"""

import logging
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.graph.state import GuardianCheck, LeadState
from src.models.watsonx_client import get_text_model

logger = logging.getLogger(__name__)

# Granite Guardian uses a specific Granite chat template
_GUARDIAN_SYSTEM = (
    "You are Granite Guardian, a safety and risk assessment AI. "
    "Your role is to identify if the assistant's response contains any of the following risks: "
    "social_bias, profanity, violence, sexual_content, unethical_behavior, groundedness, relevance."
)

_RISK_PATTERN = re.compile(r"\b(unsafe|safe)\b", re.IGNORECASE)
_RISK_TYPE_PATTERN = re.compile(
    r"\b(social_bias|profanity|violence|sexual_content|unethical_behavior|groundedness|relevance)\b",
    re.IGNORECASE,
)


def _format_guardian_prompt(user_content: str, assistant_content: str) -> str:
    """
    Granite Guardian uses the Granite chat template with a specific system prompt.
    The model returns 'safe' or 'unsafe' followed by an optional risk label.
    """
    return (
        f"<|start_of_role|>system<|end_of_role|>{_GUARDIAN_SYSTEM}<|end_of_text|>\n"
        f"<|start_of_role|>user<|end_of_role|>{user_content}<|end_of_text|>\n"
        f"<|start_of_role|>assistant<|end_of_role|>{assistant_content}<|end_of_text|>\n"
        f"<|start_of_role|>safety<|end_of_role|>"
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def _call_guardian(user_content: str, assistant_content: str) -> tuple[bool, str | None, str]:
    """Call Guardian model and parse output. Returns (is_safe, risk_type, raw_output)."""
    model = get_text_model(settings.model_guardian)
    prompt = _format_guardian_prompt(user_content, assistant_content)

    response = model.generate(prompt=prompt)
    raw = response["results"][0]["generated_text"].strip().lower()

    is_safe = not raw.startswith("unsafe")
    risk_match = _RISK_TYPE_PATTERN.search(raw)
    risk_type = risk_match.group(0) if risk_match else None

    return is_safe, risk_type, raw


def _get_content_to_check(state: LeadState, stage: str) -> tuple[str, str]:
    """Return (user_context, assistant_output) tuple for the given stage."""
    user_context = f"Analysing lead for {state['company_name']} ({state.get('contact_title', 'Unknown role')})"

    if stage == "qualification":
        assistant_output = state.get("qualification_rationale") or ""
    elif stage == "outreach":
        body = state.get("outreach_body") or ""
        subject = state.get("outreach_subject") or ""
        assistant_output = f"Subject: {subject}\n\n{body}"
    else:
        assistant_output = ""

    return user_context, assistant_output


def run_guardian_check(stage: str):
    """
    Factory that returns a LangGraph node function for the given pipeline stage.
    Usage: graph.add_node("guardian_qualification", run_guardian_check("qualification"))
    """
    def _node(state: LeadState) -> dict[str, Any]:
        user_ctx, assistant_out = _get_content_to_check(state, stage)

        if not assistant_out.strip():
            # Nothing to check — pass through
            check: GuardianCheck = {
                "stage": stage,
                "safe": True,
                "risk_type": None,
                "reason": "No content to check",
                "raw_output": "",
            }
            return {
                "current_step": f"guardian_{stage}_skipped",
                "guardian_checks": [check],
            }

        logger.info(
            "Guardian check started",
            extra={"lead_id": state["lead_id"], "stage": stage},
        )

        retries_used = 0
        max_retries = settings.guardian_max_retries
        final_safe = True
        final_risk = None
        final_raw = ""

        for attempt in range(max_retries + 1):
            try:
                is_safe, risk_type, raw = _call_guardian(user_ctx, assistant_out)
                final_safe = is_safe
                final_risk = risk_type
                final_raw = raw

                if is_safe:
                    break
                retries_used = attempt + 1
                logger.warning(
                    "Guardian flagged content",
                    extra={
                        "lead_id": state["lead_id"],
                        "stage": stage,
                        "risk_type": risk_type,
                        "attempt": attempt + 1,
                    },
                )

            except Exception as exc:
                logger.error(
                    "Guardian call failed",
                    extra={"lead_id": state["lead_id"], "error": str(exc)},
                )
                # Fail open on Guardian errors (don't block the pipeline)
                final_safe = True
                final_raw = f"guardian_error: {exc}"
                break

        check: GuardianCheck = {
            "stage": stage,
            "safe": final_safe,
            "risk_type": final_risk,
            "reason": final_raw,
            "raw_output": final_raw,
        }

        requires_review = not final_safe
        flags = [f"{stage}:{final_risk}"] if (not final_safe and final_risk) else []

        logger.info(
            "Guardian check complete",
            extra={
                "lead_id": state["lead_id"],
                "stage": stage,
                "safe": final_safe,
                "human_review": requires_review,
            },
        )

        return {
            "current_step": f"guardian_{stage}_{'passed' if final_safe else 'flagged'}",
            "guardian_checks": [check],
            "safety_passed": final_safe,
            "human_review_required": requires_review or state.get("human_review_required", False),
            "safety_flags": flags,
        }

    _node.__name__ = f"guardian_{stage}"
    return _node


# Pre-built stage-specific Guardian nodes
guardian_qualification_node = run_guardian_check("qualification")
guardian_outreach_node = run_guardian_check("outreach")
