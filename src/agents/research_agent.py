"""
Research Agent — enriches a lead record with publicly available company intelligence.

Uses web search + LLM extraction to populate:
  - company_description, employee_count, annual_revenue_estimate
  - funding_stage, tech_stack, pain_points, recent_news, company_location
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.graph.state import LeadState
from src.models.watsonx_client import get_fast_model
from src.tools.web_search import company_research_queries, fetch_page_text, web_search

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a B2B sales intelligence analyst. Given raw web search snippets about a company, extract structured information.

Return ONLY valid JSON with exactly these keys (use null if unknown):
{
  "company_description": "1-2 sentence description of what the company does",
  "employee_count": "e.g. '200-500' or '1,200'",
  "annual_revenue_estimate": "e.g. '$10M-$50M ARR' or null",
  "funding_stage": "e.g. 'Series B', 'Bootstrapped', 'Public' or null",
  "tech_stack": ["list", "of", "technologies"],
  "pain_points": ["list of inferred business challenges"],
  "recent_news": ["list of 1-3 recent noteworthy events"],
  "company_location": "City, Country"
}

Be concise and factual. Do NOT hallucinate. If data is absent, use null or [].
"""


def _build_research_context(company_name: str, website: str | None) -> str:
    queries = company_research_queries(company_name, website)
    snippets: list[str] = []

    for query in queries:
        results = web_search(query, max_results=3)
        for r in results:
            snippets.append(f"[{r.title}] {r.snippet}")

    if website:
        page_text = fetch_page_text(website, max_chars=3000)
        if page_text:
            snippets.append(f"[Company website] {page_text}")

    if not snippets:
        return f"No web data found for {company_name}."

    combined = "\n\n".join(snippets[:12])  # cap at 12 snippets to stay within context
    return combined[:6000]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _extract_with_llm(context: str, company_name: str) -> dict[str, Any]:
    llm = get_fast_model()
    user_msg = (
        f"Company: {company_name}\n\n"
        f"Web Research Data:\n{context}\n\n"
        "Extract structured information as JSON."
    )
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    raw = response.content.strip()

    # Extract JSON block robustly
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def run_research_agent(state: LeadState) -> dict[str, Any]:
    """LangGraph node: enrich the lead with company research."""
    company_name = state["company_name"]
    website = state.get("company_website")

    logger.info(
        "Research agent started",
        extra={"lead_id": state["lead_id"], "company": company_name},
    )

    try:
        context = _build_research_context(company_name, website)
        extracted = _extract_with_llm(context, company_name)

        return {
            "current_step": "research_complete",
            "company_description": extracted.get("company_description"),
            "employee_count": extracted.get("employee_count"),
            "annual_revenue_estimate": extracted.get("annual_revenue_estimate"),
            "funding_stage": extracted.get("funding_stage"),
            "tech_stack": extracted.get("tech_stack") or [],
            "pain_points": extracted.get("pain_points") or [],
            "recent_news": extracted.get("recent_news") or [],
            "company_location": extracted.get("company_location"),
        }

    except Exception as exc:
        logger.error(
            "Research agent failed",
            extra={"lead_id": state["lead_id"], "error": str(exc)},
        )
        return {
            "current_step": "research_failed",
            "errors": [f"Research agent error: {exc}"],
            "company_description": None,
            "employee_count": None,
            "annual_revenue_estimate": None,
            "funding_stage": None,
            "tech_stack": [],
            "pain_points": [],
            "recent_news": [],
            "company_location": None,
        }
