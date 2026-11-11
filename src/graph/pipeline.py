"""
SalesMind AI LangGraph pipeline.

Graph topology:
    START
      │
      ▼
  [research]
      │
      ▼
  [qualification]
      │
      ▼
  [guardian_qualification]
      │ conditional edge
      ├─ qualified + safe  ──►  [outreach]
      │                              │
      │                              ▼
      │                     [guardian_outreach]
      │                              │ conditional edge
      │                    ┌─────────┴──────────┐
      │                    ▼                    ▼
      ├─ not qualified     [crm_update]    [human_review]
      │       │                │                │
      ▼       ▼                ▼                ▼
  [human_review]             END              END
        │
        ▼
       END
"""

import logging
import os
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import settings
from src.agents.crm_agent import run_crm_agent
from src.agents.outreach_agent import run_outreach_agent
from src.agents.qualification_agent import run_qualification_agent
from src.agents.research_agent import run_research_agent
from src.graph.router import (
    route_after_outreach_guardian,
    route_after_qualification_guardian,
)
from src.graph.state import LeadState, initial_state
from src.guardrails.guardian import guardian_outreach_node, guardian_qualification_node

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Human-review terminal node
# --------------------------------------------------------------------------- #

def human_review_node(state: LeadState) -> dict[str, Any]:
    """Terminal node: flag the lead for manual review and build a partial CRM record."""
    logger.warning(
        "Lead flagged for human review",
        extra={
            "lead_id": state["lead_id"],
            "flags": state.get("safety_flags"),
            "score": state.get("qualification_score"),
        },
    )
    # Still build a CRM record so the lead isn't lost
    from datetime import datetime, timezone
    return {
        "current_step": "human_review_required",
        "pipeline_end_time": datetime.now(timezone.utc).isoformat(),
        "crm_status": "pending_human_review",
    }


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_pipeline(use_checkpointing: bool = True) -> StateGraph:
    """Build and compile the LangGraph StateGraph."""

    graph = StateGraph(LeadState)

    # Nodes
    graph.add_node("research", run_research_agent)
    graph.add_node("qualification", run_qualification_agent)
    graph.add_node("guardian_qualification", guardian_qualification_node)
    graph.add_node("outreach", run_outreach_agent)
    graph.add_node("guardian_outreach", guardian_outreach_node)
    graph.add_node("crm_update", run_crm_agent)
    graph.add_node("human_review", human_review_node)

    # Linear edges
    graph.add_edge(START, "research")
    graph.add_edge("research", "qualification")
    graph.add_edge("qualification", "guardian_qualification")

    # Conditional: after qualification guardian
    graph.add_conditional_edges(
        "guardian_qualification",
        route_after_qualification_guardian,
        {
            "outreach": "outreach",
            "crm_update": "crm_update",
            "human_review": "human_review",
        },
    )

    graph.add_edge("outreach", "guardian_outreach")

    # Conditional: after outreach guardian
    graph.add_conditional_edges(
        "guardian_outreach",
        route_after_outreach_guardian,
        {
            "crm_update": "crm_update",
            "human_review": "human_review",
        },
    )

    graph.add_edge("crm_update", END)
    graph.add_edge("human_review", END)

    # Compile (with optional SQLite checkpointing for persistence)
    if use_checkpointing:
        os.makedirs("checkpoints", exist_ok=True)
        memory = SqliteSaver.from_conn_string(settings.langgraph_checkpoint_uri)
        return graph.compile(checkpointer=memory)

    return graph.compile()


# Singleton compiled pipeline (lazy-initialised)
_pipeline = None


def get_pipeline(use_checkpointing: bool = True):
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline(use_checkpointing=use_checkpointing)
    return _pipeline


# --------------------------------------------------------------------------- #
# Public run function
# --------------------------------------------------------------------------- #

def run_pipeline(lead_data: dict[str, Any], thread_id: str | None = None) -> dict[str, Any]:
    """
    Run the full SalesMind AI pipeline for a single lead.

    Args:
        lead_data: Raw lead dictionary (see LeadState for required fields).
        thread_id: Optional LangGraph thread ID for checkpointed resumption.

    Returns:
        Final LeadState as a plain dict.
    """
    state = initial_state(lead_data)
    pipeline = get_pipeline()

    config = {"configurable": {"thread_id": thread_id or state["lead_id"]}}

    logger.info(
        "Pipeline started",
        extra={"lead_id": state["lead_id"], "company": state["company_name"]},
    )

    try:
        final_state = pipeline.invoke(state, config=config)
        logger.info(
            "Pipeline complete",
            extra={
                "lead_id": final_state["lead_id"],
                "qualified": final_state.get("is_qualified"),
                "score": final_state.get("qualification_score"),
                "step": final_state.get("current_step"),
            },
        )
        return dict(final_state)

    except Exception as exc:
        logger.error(
            "Pipeline crashed",
            extra={"lead_id": state["lead_id"], "error": str(exc)},
        )
        raise


async def arun_pipeline(lead_data: dict[str, Any], thread_id: str | None = None) -> dict[str, Any]:
    """Async version of run_pipeline for use with FastAPI."""
    import asyncio
    return await asyncio.to_thread(run_pipeline, lead_data, thread_id)
