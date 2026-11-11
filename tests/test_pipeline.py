"""
Integration tests for the full LangGraph pipeline.
All LLM/Guardian calls are mocked — no real credentials required.
"""

import json
import pytest
from unittest.mock import MagicMock, patch


def _make_qualification_response(score: int = 75, qualified: bool = True):
    mock = MagicMock()
    mock.content = json.dumps({
        "bant_score": {
            "budget": score // 4,
            "authority": score // 4,
            "need": score // 4,
            "timeline": score - 3 * (score // 4),
            "total": score,
        },
        "confidence": 0.85,
        "rationale": "Test rationale",
        "primary_strength": "Budget confirmed",
        "primary_weakness": "Timeline unclear",
    })
    return mock


def _make_outreach_response():
    mock = MagicMock()
    mock.content = json.dumps({
        "subject": "Quick question about scaling challenges",
        "body": "Hi Jane,\n\nI noticed your recent Series B — congrats!\n\nBest,\nAlex",
        "channel": "email",
        "tone": "conversational",
    })
    return mock


def _make_research_response():
    mock = MagicMock()
    mock.content = json.dumps({
        "company_description": "Test company description.",
        "employee_count": "200-500",
        "annual_revenue_estimate": "$20M-$50M",
        "funding_stage": "Series B",
        "tech_stack": ["Python", "AWS"],
        "pain_points": ["scalability"],
        "recent_news": ["Series B raised"],
        "company_location": "San Francisco, USA",
    })
    return mock


class TestPipelineIntegration:
    """Tests that the full graph runs end-to-end with mocked LLMs."""

    @pytest.fixture(autouse=True)
    def _patch_llms(self):
        """Patch all LLM and Guardian calls for the entire test class."""
        qual_llm = MagicMock()
        qual_llm.invoke.return_value = _make_qualification_response(score=78, qualified=True)

        outreach_llm = MagicMock()
        outreach_llm.invoke.return_value = _make_outreach_response()

        research_llm = MagicMock()
        research_llm.invoke.return_value = _make_research_response()

        guardian_model = MagicMock()
        guardian_model.generate.return_value = {
            "results": [{"generated_text": "safe"}]
        }

        with (
            patch("src.agents.research_agent.web_search", return_value=[]),
            patch("src.agents.research_agent.fetch_page_text", return_value=""),
            patch("src.agents.research_agent.get_fast_model", return_value=research_llm),
            patch("src.agents.qualification_agent.get_chat_model", return_value=qual_llm),
            patch("src.agents.outreach_agent.get_deep_model", return_value=outreach_llm),
            patch("src.guardrails.guardian.get_text_model", return_value=guardian_model),
            patch("src.graph.pipeline.SqliteSaver") as mock_saver,
        ):
            mock_saver.from_conn_string.return_value = MagicMock()
            yield

    def test_qualified_lead_full_pipeline(self, qualified_lead_data):
        from src.graph.pipeline import run_pipeline
        result = run_pipeline(qualified_lead_data)

        assert result["is_qualified"] is True
        assert result["qualification_score"] == 78
        assert result["outreach_body"] is not None
        assert result["crm_status"] == "created"
        assert result["human_review_required"] is False

    def test_unqualified_lead_skips_outreach(self, unqualified_lead_data):
        from src.agents.qualification_agent import run_qualification_agent
        from src.graph.state import initial_state

        # Override qualification mock to return low score
        low_score_llm = MagicMock()
        low_score_llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "bant_score": {"budget": 2, "authority": 2, "need": 3, "timeline": 1, "total": 8},
                "confidence": 0.92,
                "rationale": "No BANT signals.",
                "primary_strength": "None",
                "primary_weakness": "Everything",
            })
        )
        with patch("src.agents.qualification_agent.get_chat_model", return_value=low_score_llm):
            state = initial_state(unqualified_lead_data)
            result = run_qualification_agent(state)

        assert result["is_qualified"] is False
        assert result["qualification_score"] == 8

    def test_pipeline_result_has_required_fields(self, qualified_lead_data):
        from src.graph.pipeline import run_pipeline
        result = run_pipeline(qualified_lead_data)

        required = [
            "lead_id", "company_name", "contact_name", "contact_email",
            "qualification_score", "is_qualified", "bant_score",
            "outreach_body", "crm_record", "guardian_checks",
            "safety_passed", "pipeline_start_time",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_crm_record_reflects_pipeline_results(self, qualified_lead_data):
        from src.graph.pipeline import run_pipeline
        result = run_pipeline(qualified_lead_data)

        crm = result["crm_record"]
        assert crm["qualification"]["score"] == result["qualification_score"]
        assert crm["contact"]["email"] == qualified_lead_data["contact_email"]
        assert crm["company"]["name"] == qualified_lead_data["company_name"]


class TestDynamicRouter:
    def test_enterprise_signal_triggers_deep_model(self):
        from src.graph.state import initial_state
        from src.graph.router import select_model
        from config.settings import settings

        state = initial_state({
            "lead_id": "test-enterprise",
            "company_name": "Mega Corp",
            "contact_name": "Alice",
            "contact_email": "alice@mega.com",
            "contact_title": "Director",
        })
        state["company_size_hint"] = "Fortune 500 global enterprise"
        state["funding_stage"] = "Public company NYSE"

        chosen = select_model(state)
        assert chosen == settings.model_deep

    def test_ceo_triggers_deep_model(self):
        from src.graph.state import initial_state
        from src.graph.router import select_model
        from config.settings import settings

        state = initial_state({
            "lead_id": "test-ceo",
            "company_name": "Startup Inc",
            "contact_name": "Bob",
            "contact_email": "bob@startup.com",
            "contact_title": "CEO",
        })
        chosen = select_model(state)
        assert chosen == settings.model_deep

    def test_regular_contact_uses_fast_model(self):
        from src.graph.state import initial_state
        from src.graph.router import select_model
        from config.settings import settings

        state = initial_state({
            "lead_id": "test-regular",
            "company_name": "Mid Co",
            "contact_name": "Carol",
            "contact_email": "carol@midco.com",
            "contact_title": "Sales Manager",
        })
        chosen = select_model(state)
        assert chosen == settings.model_fast
