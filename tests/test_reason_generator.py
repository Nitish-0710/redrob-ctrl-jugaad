"""
tests/test_reason_generator.py
==============================
Unit tests for ReasonGenerator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.feature_extractor import FeatureVector
from src.scoring.scoring_engine import ScoreCard
from src.explainability.reason_generator import ReasonGenerator, ConfidenceLevel

def _make_ideal() -> tuple[FeatureVector, ScoreCard]:
    fv = FeatureVector(
        candidate_id="CAND_IDEAL",
        retrieval_skill_count=3,
        retrieval_evidence_count=4,
        ranking_skill_count=2,
        ranking_evidence_count=2,
        product_ml_experience_years=5.0,
        years_experience=7.0,
        product_company_ratio=1.0,
        github_activity_score=80.0,
        recruiter_response_rate=0.9,
        interview_completion_rate=0.9,
        honeypot_confidence=0.1,
        notice_period_days=30
    )
    sc = ScoreCard(
        candidate_id="CAND_IDEAL",
        technical_fit_score=90.0,
        career_fit_score=90.0,
        behavioral_score=90.0,
        availability_score=100.0,
        evidence_score=90.0,
        company_quality_score=90.0,
        skill_quality_score=80.0,
        honeypot_penalty=0.1,
        final_score=85.0
    )
    return fv, sc

def test_ideal_candidate_strengths():
    fv, sc = _make_ideal()
    generator = ReasonGenerator()
    result = generator.generate(fv, sc)
    
    assert "Strong retrieval and semantic search experience" in result.strengths
    assert "Meaningful product-company ML background" in result.strengths
    assert any("Active GitHub profile" in s for s in result.strengths)
    assert len(result.concerns) == 0
    assert result.confidence_level == ConfidenceLevel.HIGH.value
    assert len(result.reasoning_text) <= 300

def test_honeypot_candidate():
    fv = FeatureVector(
        candidate_id="CAND_TRAP",
        honeypot_confidence=0.95,
        years_experience=5.0
    )
    sc = ScoreCard(
        candidate_id="CAND_TRAP",
        technical_fit_score=50.0,
        career_fit_score=50.0,
        behavioral_score=50.0,
        availability_score=50.0,
        evidence_score=50.0,
        company_quality_score=50.0,
        skill_quality_score=50.0,
        honeypot_penalty=1.0,
        final_score=0.0
    )
    
    generator = ReasonGenerator()
    result = generator.generate(fv, sc)
    
    assert "Profile flagged as highly suspicious (honeypot candidate)" in result.concerns
    assert result.confidence_level == ConfidenceLevel.LOW.value
    assert "Rejected due to suspicious profile signals." in result.summary_reason
    assert "profile flagged — contradictory signals detected" in result.reasoning_text

def test_services_candidate_concerns():
    fv = FeatureVector(
        candidate_id="CAND_SERVICES",
        product_company_ratio=0.1,
        years_experience=8.0,
        notice_period_days=90,
        recruiter_response_rate=0.2,
        honeypot_confidence=0.0
    )
    sc = ScoreCard(
        candidate_id="CAND_SERVICES",
        technical_fit_score=40.0,
        career_fit_score=40.0,
        behavioral_score=40.0,
        availability_score=40.0,
        evidence_score=20.0,
        company_quality_score=10.0,
        skill_quality_score=40.0,
        honeypot_penalty=0.0,
        final_score=35.0
    )
    
    generator = ReasonGenerator()
    result = generator.generate(fv, sc)
    
    assert "Primarily services-company experience" in result.concerns
    assert "Extended notice period" in result.concerns
    assert "Low recruiter responsiveness" in result.concerns
    assert "Limited evidence of production retrieval systems" in result.concerns
    
    assert result.confidence_level == ConfidenceLevel.MEDIUM.value
    assert "limited retrieval evidence" in result.reasoning_text

def test_reasoning_text_truncation():
    fv, sc = _make_ideal()
    # Add many strengths to force a long string
    fv.vector_db_skill_count = 5
    fv.product_company_ratio = 1.0
    
    generator = ReasonGenerator()
    result = generator.generate(fv, sc)
    
    assert len(result.reasoning_text) <= 300

def test_batch_generation():
    fv, sc = _make_ideal()
    generator = ReasonGenerator()
    results = generator.generate_batch([fv, fv], [sc, sc])
    assert len(results) == 2
    assert results[0].candidate_id == "CAND_IDEAL"

if __name__ == "__main__":
    test_classes = [
        test_ideal_candidate_strengths,
        test_honeypot_candidate,
        test_services_candidate_concerns,
        test_reasoning_text_truncation,
        test_batch_generation
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
