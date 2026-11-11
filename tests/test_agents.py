"""
Unit tests for individual agents.
All Watsonx/LLM calls are mocked — no real credentials required.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.graph.state import initial_state


# --------------------------------------------------------------------------- #
# Research Agent
# --------------------------------------------------------------------------- #

class TestResearchAgent:
    def test_successful_research(self, qualified_lead_data, mock_research_llm):
        with (
            patch("src.agents.research_agent.web_search", return_value=[]),
            patch("src.agents.research_agent.fetch_page_text", return_value=""),
            patch("src.agents.research_agent.get_fast_model", return_value=mock_research_llm),
        ):
            from src.agents.research_agent import run_research_agent
            state = initial_state(qualified_lead_data)
            result = run_research_agent(state)

        assert result["current_step"] == "research_complete"
        assert result["company_description"] == "Acme Corp builds SaaS tools."
        assert result["funding_stage"] == "Series B"
        assert isinstance(result["tech_stack"], list)

    def test_research_handles_llm_failure(self, qualified_lead_data):
        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = RuntimeError("LLM timeout")

        with (
            patch("src.agents.research_agent.web_search", return_value=[]),
            patch("src.agents.research_agent.fetch_page_text", return_value=""),
            patch("src.agents.research_agent.get_fast_model", return_value=failing_llm),
        ):
            from src.agents.research_agent import run_research_agent
            state = initial_state(qualified_lead_data)
            result = run_research_agent(state)

        # Should fail gracefully, not raise
        assert result["current_step"] == "research_failed"
        assert len(result["errors"]) > 0
        assert result["company_description"] is None


# --------------------------------------------------------------------------- #
# Qualification Agent
# --------------------------------------------------------------------------- #

class TestQualificationAgent:
    def test_qualifies_strong_lead(self, qualified_state, mock_qualification_llm):
        with patch("src.agents.qualification_agent.get_chat_model", return_value=mock_qualification_llm):
            from src.agents.qualification_agent import run_qualification_agent
            result = run_qualification_agent(qualified_state)

        assert result["is_qualified"] is True
        assert result["qualification_score"] == 81
        assert result["confidence"] == pytest.approx(0.88)
        assert result["bant_score"]["budget"] == 20
        assert result["bant_score"]["total"] == 81

    def test_rejects_weak_lead(self, unqualified_lead_data):
        weak_response = MagicMock()
        weak_response.content = json.dumps({
            "bant_score": {"budget": 3, "authority": 2, "need": 5, "timeline": 1, "total": 11},
            "confidence": 0.91,
            "rationale": "No budget, no authority, vague interest.",
            "primary_strength": "None",
            "primary_weakness": "No BANT signals",
        })
        weak_llm = MagicMock()
        weak_llm.invoke.return_value = weak_response

        with patch("src.agents.qualification_agent.get_chat_model", return_value=weak_llm):
            from src.agents.qualification_agent import run_qualification_agent
            state = initial_state(unqualified_lead_data)
            result = run_qualification_agent(state)

        assert result["is_qualified"] is False
        assert result["qualification_score"] == 11

    def test_bant_total_is_sum_of_components(self, qualified_state, mock_qualification_llm):
        with patch("src.agents.qualification_agent.get_chat_model", return_value=mock_qualification_llm):
            from src.agents.qualification_agent import run_qualification_agent
            result = run_qualification_agent(qualified_state)

        bant = result["bant_score"]
        assert bant["total"] == bant["budget"] + bant["authority"] + bant["need"] + bant["timeline"]


# --------------------------------------------------------------------------- #
# Outreach Agent
# --------------------------------------------------------------------------- #

class TestOutreachAgent:
    def test_generates_email_for_qualified_lead(self, qualified_state, mock_outreach_llm):
        qualified_state["is_qualified"] = True
        qualified_state["qualification_score"] = 81
        qualified_state["qualification_rationale"] = "Strong BANT signals."
        qualified_state["contact_title"] = "Director of Engineering"  # triggers email channel

        with patch("src.agents.outreach_agent.get_deep_model", return_value=mock_outreach_llm):
            from src.agents.outreach_agent import run_outreach_agent
            result = run_outreach_agent(qualified_state)

        assert result["current_step"] == "outreach_complete"
        assert result["outreach_channel"] == "email"
        assert result["outreach_subject"] is not None
        assert len(result["outreach_body"]) > 20

    def test_skips_outreach_for_unqualified(self, unqualified_lead_data):
        from src.agents.outreach_agent import run_outreach_agent
        state = initial_state(unqualified_lead_data)
        state["is_qualified"] = False

        result = run_outreach_agent(state)

        assert result["current_step"] == "outreach_skipped"
        assert result["outreach_body"] is None

    def test_linkedin_for_ceo(self, qualified_state, mock_outreach_llm):
        """C-suite contacts should get LinkedIn outreach."""
        qualified_state["is_qualified"] = True
        qualified_state["contact_title"] = "CEO"

        linkedin_response = MagicMock()
        linkedin_response.content = json.dumps({
            "subject": None,
            "body": "Hi Jane, saw Acme's Series B — would love to connect on scaling challenges.",
            "channel": "linkedin",
            "tone": "professional",
        })
        linkedin_llm = MagicMock()
        linkedin_llm.invoke.return_value = linkedin_response

        with patch("src.agents.outreach_agent.get_deep_model", return_value=linkedin_llm):
            from src.agents.outreach_agent import run_outreach_agent
            result = run_outreach_agent(qualified_state)

        assert result["outreach_channel"] == "linkedin"


# --------------------------------------------------------------------------- #
# CRM Agent
# --------------------------------------------------------------------------- #

class TestCRMAgent:
    def test_builds_complete_crm_record(self, qualified_state):
        qualified_state.update({
            "is_qualified": True,
            "qualification_score": 81,
            "bant_score": {"budget": 20, "authority": 22, "need": 21, "timeline": 18, "total": 81},
            "outreach_body": "Hi Jane...",
            "outreach_channel": "email",
        })

        from src.agents.crm_agent import run_crm_agent
        result = run_crm_agent(qualified_state)

        assert result["crm_status"] == "created"
        record = result["crm_record"]
        assert record["lead_id"] == qualified_state["lead_id"]
        assert record["contact"]["email"] == qualified_state["contact_email"]
        assert record["qualification"]["score"] == 81
        assert record["qualification"]["is_qualified"] is True
        assert record["outreach"]["body"] == "Hi Jane..."

    def test_crm_record_has_required_keys(self, qualified_state):
        from src.agents.crm_agent import run_crm_agent
        result = run_crm_agent(qualified_state)
        record = result["crm_record"]

        for key in ("lead_id", "contact", "company", "qualification", "outreach", "safety", "pipeline"):
            assert key in record, f"Missing key: {key}"
