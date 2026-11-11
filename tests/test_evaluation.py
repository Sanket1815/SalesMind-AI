"""
Tests for the evaluation system — dataset loader and metrics computation.
Does NOT run the actual pipeline; uses synthetic EvalResult objects.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch

from evaluation.dataset import EvalSample, load_dataset
from evaluation.metrics import AggregateMetrics, EvalResult, compute_metrics, format_report


# --------------------------------------------------------------------------- #
# Dataset loader tests
# --------------------------------------------------------------------------- #

class TestDatasetLoader:
    def test_loads_default_dataset(self):
        samples = load_dataset()
        assert len(samples) >= 10

    def test_sample_has_required_fields(self):
        samples = load_dataset()
        for s in samples:
            assert isinstance(s.lead, dict)
            assert "company_name" in s.lead
            assert isinstance(s.ground_truth_qualified, bool)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(path=tmp_path / "nonexistent.json")

    def test_custom_dataset_path(self, tmp_path):
        data = [
            {
                "lead": {
                    "lead_id": "t1",
                    "company_name": "Test Co",
                    "contact_name": "T",
                    "contact_email": "t@test.com",
                    "contact_title": "CEO",
                },
                "ground_truth": {
                    "is_qualified": True,
                    "expected_score_range": [70, 90],
                    "notes": "test",
                },
            }
        ]
        p = tmp_path / "test_leads.json"
        p.write_text(json.dumps(data))
        samples = load_dataset(path=p)
        assert len(samples) == 1
        assert samples[0].ground_truth_qualified is True
        assert samples[0].expected_score_range == (70, 90)


# --------------------------------------------------------------------------- #
# Metrics computation tests
# --------------------------------------------------------------------------- #

def _make_results(
    predicted: list[bool],
    truth: list[bool],
    scores: list[int] | None = None,
    latencies: list[float] | None = None,
) -> tuple[list[EvalResult], list[EvalSample]]:
    scores = scores or [50] * len(predicted)
    latencies = latencies or [2.0] * len(predicted)
    results = [
        EvalResult(
            lead_id=f"r{i}",
            predicted_qualified=p,
            ground_truth_qualified=t,
            predicted_score=s,
            expected_score_midpoint=None,
            outreach_keywords_found=[],
            outreach_keywords_missing=[],
            model_used="ibm/granite-3-3-8b-instruct",
            pipeline_latency_s=lat,
            guardian_blocked=False,
        )
        for i, (p, t, s, lat) in enumerate(zip(predicted, truth, scores, latencies))
    ]
    samples = [
        EvalSample(lead={}, ground_truth_qualified=t)
        for t in truth
    ]
    return results, samples


class TestMetricsComputation:
    def test_perfect_accuracy(self):
        results, samples = _make_results(
            predicted=[True, True, False, False],
            truth=[True, True, False, False],
        )
        m = compute_metrics(results, samples)
        assert m.accuracy == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)

    def test_all_wrong(self):
        results, samples = _make_results(
            predicted=[True, True],
            truth=[False, False],
        )
        m = compute_metrics(results, samples)
        assert m.accuracy == pytest.approx(0.0)

    def test_guardian_block_rate(self):
        results, samples = _make_results(
            predicted=[True, False, True],
            truth=[True, False, True],
        )
        results[0].guardian_blocked = True
        m = compute_metrics(results, samples)
        assert m.guardian_block_rate == pytest.approx(1 / 3)

    def test_avg_latency(self):
        results, samples = _make_results(
            predicted=[True, False],
            truth=[True, False],
            latencies=[4.0, 6.0],
        )
        m = compute_metrics(results, samples)
        assert m.avg_latency_s == pytest.approx(5.0)

    def test_model_escalation_rate(self):
        results, samples = _make_results(
            predicted=[True, True, False],
            truth=[True, True, False],
        )
        results[0].model_used = "ibm/granite-4.1-32b-instruct"
        m = compute_metrics(results, samples)
        assert m.model_escalation_rate == pytest.approx(1 / 3)

    def test_raises_on_empty_results(self):
        with pytest.raises(ValueError):
            compute_metrics([], [])


class TestFormatReport:
    def test_report_contains_key_sections(self):
        results, samples = _make_results(
            predicted=[True, False],
            truth=[True, False],
        )
        m = compute_metrics(results, samples)
        report = format_report(m)

        assert "QUALIFICATION ACCURACY" in report
        assert "OUTREACH" in report
        assert "SAFETY" in report
        assert "SYSTEM" in report
        assert "COUNTS" in report

    def test_report_is_string(self):
        results, samples = _make_results([True], [True])
        m = compute_metrics(results, samples)
        assert isinstance(format_report(m), str)
