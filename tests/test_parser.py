"""
tests/test_parser.py
====================
Smoke tests for the parser layer.
Reads the first 1,000 lines of candidates.jsonl and validates types.
"""

import sys
from pathlib import Path

# Ensure repo root is on path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import itertools
from datetime import date

from config.settings import DEFAULT_CONFIG
from src.parser.candidate_parser import parse_candidates
from src.models.candidate import (
    CandidateRecord, SkillProficiency, WorkMode
)

logging.basicConfig(level=logging.WARNING)

DATA_PATH = DEFAULT_CONFIG.pipeline  # unused; just test import
CANDIDATES_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)


def test_parse_first_1000() -> None:
    """Parse the first 1,000 candidates and run type assertions."""
    records: list[CandidateRecord] = []

    gen = parse_candidates(CANDIDATES_FILE, progress_every=500)
    for rec in itertools.islice(gen, 1000):
        records.append(rec)

    assert len(records) > 0, "No records parsed"
    print(f"Parsed {len(records)} records")

    for rec in records:
        # Identity
        assert rec.candidate_id.startswith("CAND_"), f"Bad ID: {rec.candidate_id}"

        # Profile fields
        assert isinstance(rec.profile.years_of_experience, float)
        assert 0 <= rec.profile.years_of_experience <= 50

        # Career
        for role in rec.career_history:
            assert isinstance(role.start_date, date)
            assert isinstance(role.description, str)

        # Skills
        for sk in rec.skills:
            assert isinstance(sk.proficiency, SkillProficiency)
            assert sk.duration_months >= 0

        # Signals
        s = rec.signals
        assert 0 <= s.recruiter_response_rate <= 1
        assert isinstance(s.last_active_date, date)
        assert isinstance(s.preferred_work_mode, WorkMode)
        assert s.github_activity_score >= -1
        assert 0 <= s.interview_completion_rate <= 1

        # Derived fields
        assert isinstance(rec.career_text, str)
        assert isinstance(rec.current_title_lower, str)
        assert isinstance(rec.days_since_active, int)

    print("✅ All assertions passed")

    # Spot-check first record
    r0 = records[0]
    print(f"\nFirst record: {r0.candidate_id}")
    print(f"  Title:   {r0.profile.current_title}")
    print(f"  YOE:     {r0.total_yoe}")
    print(f"  Skills:  {len(r0.skills)}")
    print(f"  Signals: notice={r0.signals.notice_period_days}d, "
          f"response_rate={r0.signals.recruiter_response_rate:.2f}")
    print(f"  Career text (truncated): {r0.career_text[:120]}...")


if __name__ == "__main__":
    test_parse_first_1000()
