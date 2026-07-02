"""
tests/test_candidate_features.py
Unit tests for src/features/candidate_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parser import parse_candidate
from src.data.validators import validate_candidate
from src.features.candidate_features import extract_features, FeatureVector

MINIMAL_RAW = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Test",
        "headline": "Senior NLP Engineer",
        "summary": "Building search engine and retrieval systems.",
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
        "description": "Learning to rank, recommendation systems.",
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
        {"name": "Communication", "proficiency": "expert", "endorsements": 100, "duration_months": 24},
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


def test_extract_features():
    c = parse_candidate(MINIMAL_RAW)
    val = validate_candidate(c)
    fv = extract_features(c, val)

    assert isinstance(fv, FeatureVector)
    assert fv.candidate_id == "CAND_0000001"
    
    # Check experience
    assert fv.years_experience == 5.0
    assert fv.total_jobs == 1
    assert fv.avg_job_duration_months == 30.0
    
    # Check skills (Python, PyTorch are AI core, Communication is not)
    assert fv.total_skill_count == 3
    assert fv.ai_skill_count == 2
    assert fv.expert_skill_count == 2
    
    # Check education
    assert fv.education_count == 1
    assert fv.tier1_count == 1
    assert fv.cs_degree_flag == 1
    
    # Check evidence (should match 'nlp', 'search', 'retrieval', 'recommendation')
    assert fv.evidence_nlp > 0
    assert fv.evidence_search > 0
    assert fv.evidence_retrieval > 0
    assert fv.evidence_recommendation > 0
    
    # Check dict conversion
    d = fv.to_dict()
    assert d["candidate_id"] == "CAND_0000001"
    assert d["ai_skill_count"] == 2


if __name__ == "__main__":
    test_extract_features()
    print("test_extract_features ... PASS")
    print("\nAll feature extraction tests PASSED.")
