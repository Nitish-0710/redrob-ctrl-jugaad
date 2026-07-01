"""
tests/test_ranker.py
====================
Unit tests for the Ranker and its min-heap memory bounding.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.candidate import CandidateRecord, CandidateProfile, RedrobSignals
from src.ranking.ranker import Ranker

def _make_dummy_candidate(cid: str) -> CandidateRecord:
    from datetime import date
    return CandidateRecord(
        candidate_id=cid,
        profile=CandidateProfile(anonymized_name="A", headline="", summary="", location="", country="", years_of_experience=5.0, current_title="Dev", current_company="", current_company_size="S", current_industry=""),
        career_history=[],
        education=[],
        skills=[],
        certifications=[],
        languages=[],
        signals=RedrobSignals(
            profile_completeness_score=50.0,
            signup_date=date(2025,1,1),
            last_active_date=date(2026,1,1),
            open_to_work_flag=True,
            profile_views_received_30d=0,
            applications_submitted_30d=0,
            recruiter_response_rate=0.5,
            avg_response_time_hours=24.0,
            skill_assessment_scores={},
            connection_count=0,
            endorsements_received=0,
            notice_period_days=30,
            expected_salary_range_inr_lpa=None,
            preferred_work_mode="REMOTE",
            willing_to_relocate=False,
            github_activity_score=-1.0,
            search_appearance_30d=0,
            saved_by_recruiters_30d=0,
            interview_completion_rate=0.5,
            offer_acceptance_rate=0.5,
            verified_email=True,
            verified_phone=False,
            linkedin_connected=False
        )
    )

def test_ranker_memory_bound():
    """Ranker should process N items but strictly return Top K."""
    stream = (_make_dummy_candidate(f"C_{i}") for i in range(1, 151))
    
    ranker = Ranker()
    # Process 150 items, but only keep 100
    top_k = ranker.rank_candidates(stream, top_k=100)
    
    assert len(top_k) == 100
    
    # Ranks should be 1 to 100
    for i, candidate in enumerate(top_k, start=1):
        assert candidate.rank == i
        
    # Scores should be descending
    for i in range(len(top_k) - 1):
        assert top_k[i].final_score >= top_k[i+1].final_score

if __name__ == "__main__":
    passed = failed = 0
    test_funcs = [
        test_ranker_memory_bound
    ]
    
    for f in test_funcs:
        try:
            f()
            print(f"  ✅ {f.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {f.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 {f.__name__}: {e}")
            failed += 1
            
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    import sys; sys.exit(0 if failed == 0 else 1)
