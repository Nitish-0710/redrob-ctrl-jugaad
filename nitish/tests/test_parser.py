"""
tests/test_parser.py
Unit tests for src/data/parser.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parser import parse_candidate, Candidate, SalaryRange, RedrobSignals


MINIMAL_RAW = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Test User",
        "headline": "Test Headline",
        "summary": "A summary.",
        "location": "Mumbai",
        "country": "India",
        "years_of_experience": 5.0,
        "current_title": "ML Engineer",
        "current_company": "TCS",
        "current_company_size": "10001+",
        "current_industry": "IT Services",
    },
    "career_history": [{
        "company": "TCS", "title": "ML Engineer",
        "start_date": "2021-01-01", "end_date": None,
        "duration_months": 30, "is_current": True,
        "industry": "IT Services", "company_size": "10001+",
        "description": "Did ML things.",
    }],
    "education": [{
        "institution": "IIT Bombay", "degree": "B.Tech",
        "field_of_study": "Computer Science",
        "start_year": 2015, "end_year": 2019,
        "grade": "9.0 CGPA", "tier": "tier_1"
    }],
    "skills": [
        {"name": "Python", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 12, "duration_months": 24},
    ],
    "certifications": [],
    "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 85.0,
        "signup_date": "2025-01-01",
        "last_active_date": "2026-06-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 15,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.72,
        "avg_response_time_hours": 12.5,
        "skill_assessment_scores": {"Python": 88.0},
        "connection_count": 350,
        "endorsements_received": 37,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 25.0, "max": 45.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 72.0,
        "search_appearance_30d": 120,
        "saved_by_recruiters_30d": 5,
        "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.8,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
}


def test_parse_returns_candidate():
    c = parse_candidate(MINIMAL_RAW)
    assert isinstance(c, Candidate)
    assert c.candidate_id == "CAND_0000001"


def test_implied_experience():
    c = parse_candidate(MINIMAL_RAW)
    assert c.implied_experience_years == 2.5


def test_salary_not_inverted():
    c = parse_candidate(MINIMAL_RAW)
    assert not c.redrob_signals.expected_salary.is_inverted


def test_has_github():
    c = parse_candidate(MINIMAL_RAW)
    assert c.redrob_signals.has_github


def test_skill_names():
    c = parse_candidate(MINIMAL_RAW)
    assert "Python" in c.skill_names
    assert "PyTorch" in c.skill_names


def test_highest_edu_tier():
    c = parse_candidate(MINIMAL_RAW)
    assert c.highest_edu_tier == "tier_1"


def test_missing_fields_graceful():
    """Parser should not raise on missing optional fields."""
    raw = {"candidate_id": "CAND_0000002"}
    c = parse_candidate(raw)
    assert c.candidate_id == "CAND_0000002"
    assert c.n_skills == 0
    assert c.n_jobs == 0


if __name__ == "__main__":
    test_parse_returns_candidate();       print("test_parse_returns_candidate ... PASS")
    test_implied_experience();            print("test_implied_experience ... PASS")
    test_salary_not_inverted();           print("test_salary_not_inverted ... PASS")
    test_has_github();                    print("test_has_github ... PASS")
    test_skill_names();                   print("test_skill_names ... PASS")
    test_highest_edu_tier();              print("test_highest_edu_tier ... PASS")
    test_missing_fields_graceful();       print("test_missing_fields_graceful ... PASS")
    print("\nAll tests PASSED.")
