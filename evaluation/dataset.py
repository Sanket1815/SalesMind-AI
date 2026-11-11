"""
Evaluation dataset loader.

Dataset format (data/evaluation/leads.json):
[
  {
    "lead": { ...LeadState input fields... },
    "ground_truth": {
      "is_qualified": true | false,
      "expected_score_range": [60, 100],   // optional acceptable range
      "outreach_must_contain": ["pain_point_keyword"],  // optional
      "notes": "human rationale"
    }
  },
  ...
]
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DATASET_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "leads.json"


@dataclass
class EvalSample:
    lead: dict[str, Any]
    ground_truth_qualified: bool
    expected_score_range: Optional[tuple[int, int]] = None
    outreach_must_contain: list[str] = field(default_factory=list)
    notes: str = ""


def load_dataset(path: Path = _DEFAULT_DATASET_PATH) -> list[EvalSample]:
    """Load evaluation dataset from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    with path.open() as f:
        raw = json.load(f)

    samples = []
    for item in raw:
        gt = item["ground_truth"]
        rng = gt.get("expected_score_range")
        samples.append(
            EvalSample(
                lead=item["lead"],
                ground_truth_qualified=gt["is_qualified"],
                expected_score_range=tuple(rng) if rng else None,
                outreach_must_contain=gt.get("outreach_must_contain", []),
                notes=gt.get("notes", ""),
            )
        )

    logger.info("Dataset loaded", extra={"num_samples": len(samples)})
    return samples
