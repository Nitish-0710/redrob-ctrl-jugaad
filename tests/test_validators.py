"""
tests/test_validators.py
Unit tests for src/data/validators.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parser import parse_candidate
from src.data.validators import validate_candidate, ValidationResult


def _make(overrides: dict = {}) -> dict:
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test", "headline": "h", "summary": "s",
            "location": "Mumbai", "country": "India",
            "years_of_experience": 5.0,
            "current_title": "ML Engineer", "current_company": "TCS",
            "current_company_size": "10001+", "current_industry": "IT Services"
        },
        "career_history": [{"company": "TCS", "title": "ML Engineer",
            "start_date": "2021-01-01", "end_date": None,
            "duration_months": 30, "is_current": True,
            "industry": "IT Services", "company_size": "10001+", "description": "ML."}],
        "education": [], "skills": [], "certifications": [], "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 75.0, "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01", "open_to_work_flag": True,
            "profile_views_received_30d": 10, "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.5, "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {}, "connection_count": 100,
            "endorsements_received": 5, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
            "preferred_work_mode": "hybrid", "willing_to_relocate": False,
            "github_activity_score": 50.0, "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 3, "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.7, "verified_email": True,
            "verified_phone": True, "linkedin_connected": False,
        }
    }
    # Deep merge overrides
    import copy; result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and k in result:
            result[k].update(v)
        else:
            result[k] = v
    return result


def test_clean_candidate_passes():
    c = parse_candidate(_make())
    r = validate_candidate(c)
    assert r.is_valid
    assert r.n_flags == 0


def test_salary_inversion_flagged():
    raw = _make({"redrob_signals": {"expected_salary_range_inr_lpa": {"min": 50.0, "max": 20.0}}})
    c = parse_candidate(raw)
    r = validate_candidate(c)
    flag_types = [f.split(":")[0] for f in r.honeypot_flags]
    assert "HP-01_SALARY_INVERTED" in flag_types


def test_date_paradox_flagged():
    raw = _make({"redrob_signals": {
        "signup_date": "2026-01-01",
        "last_active_date": "2025-06-01"
    }})
    c = parse_candidate(raw)
    r = validate_candidate(c)
    flag_types = [f.split(":")[0] for f in r.honeypot_flags]
    assert "HP-03_DATE_PARADOX" in flag_types


def test_inflated_yoe_flagged():
    # Career is 2.5 yrs but claimed is 10 yrs
    raw = _make({"profile": {"years_of_experience": 10.0}})
    c = parse_candidate(raw)
    r = validate_candidate(c)
    flag_types = [f.split(":")[0] for f in r.honeypot_flags]
    assert "HP-04_INFLATED_YOE" in flag_types


def test_trust_score_decreases_with_flags():
    raw = _make({
        "redrob_signals": {"expected_salary_range_inr_lpa": {"min": 50.0, "max": 10.0}},
        "profile": {"years_of_experience": 20.0}
    })
    c = parse_candidate(raw)
    r = validate_candidate(c)
    assert r.trust_score < 1.0


if __name__ == "__main__":
    test_clean_candidate_passes();       print("test_clean_candidate_passes ... PASS")
    test_salary_inversion_flagged();     print("test_salary_inversion_flagged ... PASS")
    test_date_paradox_flagged();         print("test_date_paradox_flagged ... PASS")
    test_inflated_yoe_flagged();         print("test_inflated_yoe_flagged ... PASS")
    test_trust_score_decreases_with_flags(); print("test_trust_score_decreases_with_flags ... PASS")
    print("\nAll validator tests PASSED.")
