"""
Evaluation metrics for the SalesMind AI pipeline.

Computed metrics:
  Qualification:
    - accuracy, precision, recall, F1 (binary classification)
    - score_mae: mean absolute error between predicted and expected BANT score range midpoints
    - guardian_block_rate: fraction of leads blocked by Guardian

  Outreach:
    - personalization_hit_rate: fraction of outreach messages containing required keywords
    - generation_rate: fraction of qualified leads that got outreach generated

  System:
    - avg_pipeline_latency_s
    - model_escalation_rate: fraction of leads routed to deep model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    lead_id: str
    predicted_qualified: bool
    ground_truth_qualified: bool
    predicted_score: int
    expected_score_midpoint: float | None
    outreach_keywords_found: list[str]
    outreach_keywords_missing: list[str]
    model_used: str
    pipeline_latency_s: float
    guardian_blocked: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    # Qualification
    accuracy: float
    precision: float
    recall: float
    f1: float
    score_mae: float | None  # None if no expected ranges provided

    # Outreach
    personalization_hit_rate: float
    outreach_generation_rate: float

    # Safety
    guardian_block_rate: float

    # System
    avg_latency_s: float
    model_escalation_rate: float  # deep model usage rate

    # Raw counts
    total_samples: int
    qualified_count: int
    not_qualified_count: int
    errors_count: int


def compute_metrics(results: list[EvalResult], samples_with_ranges: list[Any]) -> AggregateMetrics:
    """Compute aggregate metrics from a list of EvalResult objects."""
    if not results:
        raise ValueError("No results to compute metrics from")

    y_true = [int(r.ground_truth_qualified) for r in results]
    y_pred = [int(r.predicted_qualified) for r in results]

    accuracy = accuracy_score(y_true, y_pred)
    # Use zero_division=0 to handle edge cases (all positive or all negative)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Score MAE (only where expected range given)
    mae_values = []
    for r, s in zip(results, samples_with_ranges):
        if s.expected_score_range and r.predicted_score is not None:
            midpoint = sum(s.expected_score_range) / 2
            mae_values.append(abs(r.predicted_score - midpoint))
    score_mae = float(np.mean(mae_values)) if mae_values else None

    # Outreach metrics
    qualified_results = [r for r in results if r.ground_truth_qualified]
    outreach_generated = [r for r in qualified_results if not r.guardian_blocked]
    outreach_generation_rate = (
        len(outreach_generated) / len(qualified_results) if qualified_results else 0.0
    )

    kw_hits = [
        r for r in results
        if r.outreach_keywords_missing == [] and r.outreach_keywords_found
    ]
    all_with_keywords = [r for r in results if r.outreach_keywords_found or r.outreach_keywords_missing]
    personalization_hit_rate = (
        len(kw_hits) / len(all_with_keywords) if all_with_keywords else 1.0
    )

    # Safety
    guardian_blocked = [r for r in results if r.guardian_blocked]
    guardian_block_rate = len(guardian_blocked) / len(results)

    # Latency
    avg_latency = float(np.mean([r.pipeline_latency_s for r in results]))

    # Model escalation
    deep_model_id_fragment = "32b"  # heuristic — deep model IDs contain "32b" or "4.1"
    escalated = [r for r in results if "4.1" in r.model_used or "32b" in r.model_used]
    model_escalation_rate = len(escalated) / len(results)

    return AggregateMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        score_mae=score_mae,
        personalization_hit_rate=personalization_hit_rate,
        outreach_generation_rate=outreach_generation_rate,
        guardian_block_rate=guardian_block_rate,
        avg_latency_s=avg_latency,
        model_escalation_rate=model_escalation_rate,
        total_samples=len(results),
        qualified_count=sum(y_pred),
        not_qualified_count=len(y_pred) - sum(y_pred),
        errors_count=sum(1 for r in results if r.errors),
    )


def format_report(metrics: AggregateMetrics) -> str:
    """Format metrics as a readable text report."""
    lines = [
        "=" * 60,
        "  SALESMIND AI — EVALUATION REPORT",
        "=" * 60,
        "",
        "QUALIFICATION ACCURACY",
        f"  Accuracy   : {metrics.accuracy:.3f}",
        f"  Precision  : {metrics.precision:.3f}",
        f"  Recall     : {metrics.recall:.3f}",
        f"  F1 Score   : {metrics.f1:.3f}",
        f"  Score MAE  : {metrics.score_mae:.1f}" if metrics.score_mae else "  Score MAE  : N/A",
        "",
        "OUTREACH",
        f"  Generation Rate       : {metrics.outreach_generation_rate:.1%}",
        f"  Personalization Hits  : {metrics.personalization_hit_rate:.1%}",
        "",
        "SAFETY (Granite Guardian)",
        f"  Block Rate            : {metrics.guardian_block_rate:.1%}",
        "",
        "SYSTEM",
        f"  Avg Pipeline Latency  : {metrics.avg_latency_s:.1f}s",
        f"  Deep Model Usage      : {metrics.model_escalation_rate:.1%}",
        "",
        "COUNTS",
        f"  Total Samples         : {metrics.total_samples}",
        f"  Predicted Qualified   : {metrics.qualified_count}",
        f"  Predicted Rejected    : {metrics.not_qualified_count}",
        f"  Samples with Errors   : {metrics.errors_count}",
        "=" * 60,
    ]
    return "\n".join(lines)
