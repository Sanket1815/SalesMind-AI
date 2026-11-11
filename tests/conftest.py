"""
Shared pytest fixtures.

All tests run without real IBM Watsonx credentials by mocking
the LLM and Guardian calls at the boundary.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.graph.state import initial_state


# --------------------------------------------------------------------------- #
# Sample leads
# --------------------------------------------------------------------------- #

@pytest.fixture
def qualified_lead_data():
    return {
        "lead_id": "test-qualified-001",
        "company_name": "Acme Corp",
        "contact_name": "Jane Doe",
        "contact_email": "jane@acmecorp.com",
        "contact_title": "VP of Engineering",
        "company_website": None,
        "industry": "B2B SaaS",
        "company_size_hint": "200-500 employees",
        "initial_message": "We have budget approved and need a solution by Q1.",
    }


@pytest.fixture
def unqualified_lead_data():
    return {
        "lead_id": "test-unqualified-001",
        "company_name": "Tiny Shop",
        "contact_name": "Bob Smith",
        "contact_email": "bob@tinyshop.com",
        "contact_title": "Intern",
        "company_website": None,
        "industry": "Retail",
        "company_size_hint": "1-5 employees",
        "initial_message": "Just browsing, no budget.",
    }


@pytest.fixture
def qualified_state(qualified_lead_data):
    state = initial_state(qualified_lead_data)
    state.update({
        "company_description": "Acme Corp builds SaaS tools for mid-market companies.",
        "employee_count": "200-500",
        "funding_stage": "Series B",
        "tech_stack": ["Python", "AWS", "PostgreSQL"],
        "pain_points": ["scalability", "data pipeline reliability"],
        "recent_news": ["Raised $20M Series B in Jan 2025"],
        "current_step": "research_complete",
    })
    return state


# --------------------------------------------------------------------------- #
# LLM mock factory
# --------------------------------------------------------------------------- #

@pytest.fixture
def mock_qualification_llm():
    """Returns a mock ChatWatsonx that outputs a valid BANT JSON."""
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "bant_score": {"budget": 20, "authority": 22, "need": 21, "timeline": 18, "total": 81},
        "confidence": 0.88,
        "rationale": "Strong BANT signals: confirmed budget, VP authority, clear need, Q1 timeline.",
        "primary_strength": "Explicit budget and timeline",
        "primary_weakness": "No direct decision-maker confirmation",
    })
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


@pytest.fixture
def mock_outreach_llm():
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "subject": "Solving Acme Corp's scalability challenges",
        "body": "Hi Jane,\n\nI noticed Acme Corp recently raised their Series B...\n\nBest,\nAlex",
        "channel": "email",
        "tone": "conversational",
    })
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


@pytest.fixture
def mock_research_llm():
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "company_description": "Acme Corp builds SaaS tools.",
        "employee_count": "200-500",
        "annual_revenue_estimate": "$20M-$50M",
        "funding_stage": "Series B",
        "tech_stack": ["Python", "AWS"],
        "pain_points": ["scalability"],
        "recent_news": ["Raised $20M Series B"],
        "company_location": "San Francisco, USA",
    })
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm
