"""
tests/test_scoring_engine.py
============================
Unit tests for ScoringEngine.

Evaluates how different FeatureVector archetypes translate into final
component scores and checks the honeypot decay mechanism.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.feature_extractor import FeatureVector
from src.scoring.scoring_engine import ScoringEngine, ScoreCard

def test_ideal_ml_engineer_score():
    """
    A strong ML engineer with retrieval experience and product background
    should get a high score across components.
    """
    fv = FeatureVector(
        candidate_id="CAND_IDEAL",
        years_experience=6.0,
        total_roles=3,
        avg_tenure_months=24.0,
        technical_role_ratio=1.0,
        retrieval_skill_count=4,
        ranking_skill_count=2,
        vector_db_skill_count=2,
        llm_skill_count=2,
        product_company_count=3,
        product_company_ratio=1.0,
        open_to_work=True,
        recruiter_response_rate=0.9,
        interview_completion_rate=0.95,
        github_activity_score=80.0,
        notice_period_days=30,
        avg_response_time_hours=12.0,
        retrieval_evidence_count=6,
        ranking_evidence_count=3,
        ml_production_evidence_count=4,
        product_ml_experience_years=5.0,
        expert_skill_count=5,
        advanced_skill_count=3,
        avg_skill_duration=48.0,
        has_skill_assessments=True,
        average_skill_assessment=85.0,
        honeypot_confidence=0.1
    )
    
    engine = ScoringEngine()
    scorecard = engine.score(fv)
    
    # Check component scores (0-100)
    assert scorecard.technical_fit_score > 80.0
    assert scorecard.career_fit_score > 80.0
    assert scorecard.availability_score == 100.0  # Perfect notice and response time
    assert scorecard.evidence_score == 100.0      # High evidence caps out
    assert scorecard.company_quality_score == 100.0
    assert scorecard.final_score > 80.0

def test_honeypot_decay():
    """
    A candidate with a high honeypot confidence should have their final score decimated.
    """
    fv = FeatureVector(
        candidate_id="CAND_TRAP",
        years_experience=5.0,
        retrieval_skill_count=5,
        vector_db_skill_count=5,
        product_ml_experience_years=5.0,
        honeypot_confidence=0.95  # > 0.85 hard threshold
    )
    
    engine = ScoringEngine()
    scorecard = engine.score(fv)
    
    # 0.95 -> 1.0 penalty
    assert scorecard.honeypot_penalty == 1.0
    assert scorecard.final_score == 0.0

def test_honeypot_soft_decay():
    """
    A candidate with moderate honeypot confidence gets a proportional penalty.
    """
    fv = FeatureVector(
        candidate_id="CAND_MILD_TRAP",
        years_experience=5.0,
        retrieval_skill_count=5,
        honeypot_confidence=0.40
    )
    
    engine = ScoringEngine()
    scorecard = engine.score(fv)
    
    assert scorecard.honeypot_penalty == 0.40
    # Final score should be 60% of what it would have been

def test_long_notice_period_penalty():
    """
    Notice period > 90 days imposes a severe availability penalty.
    """
    fv = FeatureVector(
        candidate_id="CAND_LONG_NOTICE",
        notice_period_days=120,
        avg_response_time_hours=100.0
    )
    
    engine = ScoringEngine()
    scorecard = engine.score(fv)
    
    # 100 - 60 (for 120 days) - 20 (for 100h response) = 20
    assert scorecard.availability_score <= 20.0

def test_junior_penalty():
    """
    < 2 YOE results in career fit penalties.
    """
    fv = FeatureVector(
        candidate_id="CAND_JUNIOR",
        years_experience=1.0,
        avg_tenure_months=12.0,
        product_ml_experience_years=0.0
    )
    
    engine = ScoringEngine()
    scorecard = engine.score(fv)
    
    # 50 - 20 (yoe penalty) = 30. No boost.
    assert scorecard.career_fit_score <= 30.0

def test_no_evidence_score():
    """
    If no evidence in descriptions, evidence score is 0.
    """
    fv = FeatureVector(candidate_id="CAND_NO_EVIDENCE")
    engine = ScoringEngine()
    scorecard = engine.score(fv)
    assert scorecard.evidence_score == 0.0

def test_score_batch():
    """Batch scoring should return correct lengths."""
    fvs = [FeatureVector(candidate_id=f"C_{i}") for i in range(5)]
    engine = ScoringEngine()
    cards = engine.score_batch(fvs)
    assert len(cards) == 5
    assert all(isinstance(c, ScoreCard) for c in cards)

def test_technical_eligibility_penalty():
    # technical_eligibility = 0
    fv = FeatureVector(
        candidate_id="CAND_NO_TECH",
        retrieval_skill_count=0,
        ranking_skill_count=0,
        vector_db_skill_count=0,
        retrieval_evidence_count=0,
        ranking_evidence_count=0,
        years_experience=6.0,
        notice_period_days=30,
        recruiter_response_rate=1.0,
        open_to_work=True
    )
    engine = ScoringEngine()
    scorecard = engine.score(fv)
    
    # Base score would be quite high due to availability and behavioral, but penalty should crush it
    assert "technical_eligibility_penalty" in scorecard.score_breakdown
    assert scorecard.final_score < 10.0  # Multiplied by 0.05

def test_non_technical_title_penalty():
    fv = FeatureVector(
        candidate_id="CAND_BIZ_ANALYST",
        current_title="Business Analyst",
        title_category="NON_TECHNICAL",
        years_experience=10.0,
        notice_period_days=30,
        recruiter_response_rate=1.0,
        open_to_work=True,
        # Give enough eligibility to pass gate 1
        retrieval_skill_count=3,
        # Make evidence below exception threshold
        product_ml_experience_years=0.0
    )
    engine = ScoringEngine()
    scorecard = engine.score(fv)
    
    assert "non_technical_penalty" in scorecard.score_breakdown
    # The score should be severely restricted
    assert scorecard.final_score < 25.0

def test_evidence_floor_penalty():
    fv = FeatureVector(
        candidate_id="CAND_NO_EVIDENCE",
        # High eligibility to pass gate 1
        retrieval_skill_count=3,
        ranking_skill_count=2,
        # No evidence
        retrieval_evidence_count=0,
        ml_production_evidence_count=0,
        years_experience=5.0,
        open_to_work=True,
        recruiter_response_rate=1.0
    )
    engine = ScoringEngine()
    scorecard = engine.score(fv)
    
    assert "low_evidence_penalty" in scorecard.score_breakdown
    assert scorecard.final_score < 50.0  # Multiplied by 0.25 since evidence < 20


if __name__ == "__main__":
    test_classes = [
        test_ideal_ml_engineer_score,
        test_honeypot_decay,
        test_honeypot_soft_decay,
        test_long_notice_period_penalty,
        test_junior_penalty,
        test_no_evidence_score,
        test_score_batch,
        test_technical_eligibility_penalty,
        test_non_technical_title_penalty,
        test_evidence_floor_penalty
    ]

    passed = failed = 0
    for method in test_classes:
        try:
            method()
            print(f"  ✅ {method.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {method.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 {method.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    import sys; sys.exit(0 if failed == 0 else 1)
