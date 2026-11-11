"""
Database seeder — populates PostgreSQL with mock pipeline results covering all scenarios.

Usage:
    python scripts/seed_db.py
    python scripts/seed_db.py --reset   # drops and recreates all data first
    python scripts/seed_db.py --file data/seeds/scenarios.json
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent.parent / "data" / "seeds" / "scenarios.json"


def _fake_timestamps(idx: int) -> tuple[str, str]:
    """Generate staggered creation times so seed data looks like real production traffic."""
    base = datetime.now(timezone.utc) - timedelta(days=30)
    start = base + timedelta(hours=idx * 7, minutes=idx * 13)
    end = start + timedelta(seconds=45 + idx * 3)
    return start.isoformat(), end.isoformat()


def _scenario_to_db_rows(scenario: dict, idx: int) -> tuple[dict, dict, list[dict]]:
    """Convert a scenario JSON entry into (lead_row, run_row, guardian_rows) dicts."""
    lead_data = scenario["lead"]
    output = scenario["pipeline_output"]
    start_time, end_time = _fake_timestamps(idx)

    lead_id = lead_data.get("lead_id") or str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    lead_row = {
        "id": uuid.UUID(lead_id) if not isinstance(lead_id, uuid.UUID) else lead_id,
        "company_name": lead_data["company_name"],
        "contact_name": lead_data["contact_name"],
        "contact_email": lead_data["contact_email"],
        "contact_title": lead_data.get("contact_title", ""),
        "company_website": lead_data.get("company_website"),
        "industry": lead_data.get("industry"),
        "company_size_hint": lead_data.get("company_size_hint"),
        "initial_message": lead_data.get("initial_message"),
        "created_at": datetime.fromisoformat(start_time),
    }

    bant = output.get("bant_score") or {}
    run_row = {
        "id": uuid.UUID(run_id),
        "lead_id": lead_row["id"],
        # Research
        "company_description": output.get("company_description"),
        "employee_count": output.get("employee_count"),
        "annual_revenue_estimate": output.get("annual_revenue_estimate"),
        "funding_stage": output.get("funding_stage"),
        "company_location": output.get("company_location"),
        "tech_stack": output.get("tech_stack") or [],
        "pain_points": output.get("pain_points") or [],
        "recent_news": output.get("recent_news") or [],
        # Qualification
        "qualification_score": output.get("qualification_score"),
        "is_qualified": output.get("is_qualified"),
        "confidence": output.get("confidence"),
        "qualification_rationale": output.get("qualification_rationale"),
        "bant_budget": bant.get("budget"),
        "bant_authority": bant.get("authority"),
        "bant_need": bant.get("need"),
        "bant_timeline": bant.get("timeline"),
        "model_used": output.get("model_used_for_qualification"),
        # Outreach
        "outreach_channel": output.get("outreach_channel"),
        "outreach_subject": output.get("outreach_subject"),
        "outreach_body": output.get("outreach_body"),
        "outreach_tone": output.get("outreach_tone"),
        # Safety
        "safety_passed": output.get("safety_passed", True),
        "human_review_required": output.get("human_review_required", False),
        "safety_flags": output.get("safety_flags") or [],
        # Metadata
        "current_step": output.get("current_step", "crm_complete"),
        "pipeline_start_time": datetime.fromisoformat(start_time),
        "pipeline_end_time": datetime.fromisoformat(end_time),
        "errors": output.get("errors") or [],
        "full_state": {**lead_data, **output},
        "created_at": datetime.fromisoformat(start_time),
    }

    guardian_rows = []
    for check in output.get("guardian_checks") or []:
        guardian_rows.append({
            "id": uuid.uuid4(),
            "run_id": uuid.UUID(run_id),
            "stage": check.get("stage", "unknown"),
            "safe": check.get("safe", True),
            "risk_type": check.get("risk_type"),
            "reason": check.get("reason"),
            "raw_output": check.get("raw_output"),
            "checked_at": datetime.fromisoformat(start_time),
        })

    return lead_row, run_row, guardian_rows


def seed(scenarios_path: Path, reset: bool = False) -> None:
    if not settings.database_url:
        logger.error(
            "DATABASE_URL is not set. Add it to your .env file:\n"
            "DATABASE_URL=postgresql+asyncpg://salesmind:secret@localhost:5432/salesmind"
        )
        sys.exit(1)

    # Seed uses sync SQLAlchemy for simplicity
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from src.db.models import Base, GuardianLog, Lead, PipelineRun

    sync_url = (
        settings.database_url
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql+psycopg2://", "postgresql://")
    )
    engine = create_engine(sync_url, echo=False)

    if reset:
        logger.info("Resetting database (dropping all tables)...")
        Base.metadata.drop_all(engine)
        logger.info("Tables dropped.")

    Base.metadata.create_all(engine)
    logger.info("Tables created/verified.")

    with open(scenarios_path) as f:
        scenarios = json.load(f)

    Session = sessionmaker(engine)
    with Session() as session:
        if reset:
            session.execute(text("DELETE FROM guardian_logs"))
            session.execute(text("DELETE FROM pipeline_runs"))
            session.execute(text("DELETE FROM leads"))
            session.commit()
            logger.info("Existing data cleared.")

        inserted_leads = 0
        inserted_runs = 0
        inserted_guardian = 0

        for idx, scenario in enumerate(scenarios):
            label = scenario.get("_scenario", f"scenario-{idx}")
            try:
                lead_row, run_row, guardian_rows = _scenario_to_db_rows(scenario, idx)

                # Skip if lead already exists
                existing = session.get(Lead, lead_row["id"])
                if existing:
                    logger.info(f"  SKIP (exists) [{label}]")
                    continue

                session.add(Lead(**lead_row))
                session.flush()

                session.add(PipelineRun(**run_row))
                session.flush()

                for g in guardian_rows:
                    session.add(GuardianLog(**g))

                session.commit()
                inserted_leads += 1
                inserted_runs += 1
                inserted_guardian += len(guardian_rows)
                logger.info(f"  OK [{label}]")

            except Exception as exc:
                session.rollback()
                logger.error(f"  FAILED [{label}]: {exc}")

    logger.info(
        f"\nSeed complete: {inserted_leads} leads, "
        f"{inserted_runs} pipeline runs, "
        f"{inserted_guardian} guardian logs inserted."
    )


def print_summary(scenarios_path: Path) -> None:
    """Print a summary of what will be seeded without touching the DB."""
    with open(scenarios_path) as f:
        scenarios = json.load(f)

    print(f"\n{'='*65}")
    print(f"  SALESMIND AI — MOCK DATA SUMMARY ({len(scenarios)} scenarios)")
    print(f"{'='*65}")

    categories = {
        "HIGHLY_QUALIFIED": [],
        "QUALIFIED": [],
        "NOT_QUALIFIED": [],
        "HUMAN_REVIEW": [],
        "DEEP_MODEL_ESCALATED": [],
        "LINKEDIN_CHANNEL": [],
        "PIPELINE_ERROR": [],
        "BORDERLINE": [],
    }
    for s in scenarios:
        label = s.get("_scenario", "")
        for cat in categories:
            if cat in label:
                categories[cat].append(s["lead"]["company_name"])
                break

    for cat, names in categories.items():
        if names:
            print(f"\n  {cat} ({len(names)}):")
            for name in names:
                print(f"    • {name}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed SalesMind AI database with mock data")
    parser.add_argument(
        "--file", type=Path, default=_DEFAULT_SCENARIOS,
        help="Path to scenarios JSON file"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear existing data before seeding"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print summary without inserting data"
    )
    args = parser.parse_args()

    print_summary(args.file)

    if args.dry_run:
        logger.info("Dry run — no data inserted.")
    else:
        seed(args.file, reset=args.reset)
