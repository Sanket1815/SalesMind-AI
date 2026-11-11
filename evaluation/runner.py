"""
Evaluation runner — runs the pipeline over the full evaluation dataset
and computes aggregate metrics.

Usage:
    python -m evaluation.runner
    python -m evaluation.runner --dataset data/evaluation/leads.json --output results/eval_2025.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

from evaluation.dataset import EvalSample, load_dataset
from evaluation.metrics import AggregateMetrics, EvalResult, compute_metrics, format_report
from src.graph.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def _run_single(sample: EvalSample) -> EvalResult:
    """Run the pipeline for one sample and return an EvalResult."""
    t0 = time.perf_counter()
    errors: list[str] = []
    final_state: dict = {}

    try:
        final_state = run_pipeline(sample.lead)
    except Exception as exc:
        errors.append(str(exc))
        logger.error("Pipeline failed for sample", extra={"error": str(exc)})

    latency = time.perf_counter() - t0

    predicted_score = final_state.get("qualification_score") or 0
    predicted_qualified = final_state.get("is_qualified") or False
    outreach_body = (final_state.get("outreach_body") or "").lower()

    keywords_found = [kw for kw in sample.outreach_must_contain if kw.lower() in outreach_body]
    keywords_missing = [kw for kw in sample.outreach_must_contain if kw.lower() not in outreach_body]

    guardian_blocked = final_state.get("human_review_required", False)

    return EvalResult(
        lead_id=final_state.get("lead_id", sample.lead.get("lead_id", "unknown")),
        predicted_qualified=predicted_qualified,
        ground_truth_qualified=sample.ground_truth_qualified,
        predicted_score=predicted_score,
        expected_score_midpoint=(
            sum(sample.expected_score_range) / 2 if sample.expected_score_range else None
        ),
        outreach_keywords_found=keywords_found,
        outreach_keywords_missing=keywords_missing,
        model_used=final_state.get("model_used_for_qualification") or "unknown",
        pipeline_latency_s=latency,
        guardian_blocked=guardian_blocked,
        errors=errors + (final_state.get("errors") or []),
    )


def run_evaluation(
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    max_samples: int | None = None,
) -> AggregateMetrics:
    """Run the full evaluation pipeline."""
    samples = load_dataset(dataset_path) if dataset_path else load_dataset()
    if max_samples:
        samples = samples[:max_samples]

    logger.info("Starting evaluation", extra={"num_samples": len(samples)})

    results: list[EvalResult] = []
    for i, sample in enumerate(samples):
        logger.info(
            "Evaluating sample",
            extra={"idx": i + 1, "total": len(samples), "company": sample.lead.get("company_name")},
        )
        result = _run_single(sample)
        results.append(result)

    metrics = compute_metrics(results, samples)
    report = format_report(metrics)
    print(report)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "score_mae": metrics.score_mae,
                "personalization_hit_rate": metrics.personalization_hit_rate,
                "outreach_generation_rate": metrics.outreach_generation_rate,
                "guardian_block_rate": metrics.guardian_block_rate,
                "avg_latency_s": metrics.avg_latency_s,
                "model_escalation_rate": metrics.model_escalation_rate,
                "total_samples": metrics.total_samples,
            },
            "individual_results": [
                {
                    "lead_id": r.lead_id,
                    "predicted_qualified": r.predicted_qualified,
                    "ground_truth_qualified": r.ground_truth_qualified,
                    "predicted_score": r.predicted_score,
                    "latency_s": r.pipeline_latency_s,
                    "guardian_blocked": r.guardian_blocked,
                    "errors": r.errors,
                }
                for r in results
            ],
        }
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Results saved", extra={"path": str(output_path)})

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run SalesMind AI evaluation")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/latest.json"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        max_samples=args.max_samples,
    )
