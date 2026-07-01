"""
tests/test_reasoning.py
=======================
Unit tests for the reasoning generator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.candidate_features import FeatureVector
from src.ranking.scorer import CandidateScore
from src.reasoning.generator import generate_reasoning
from src.data.parser import Candidate


def test_generate_reasoning():
    # Mock FeatureVector
    fv = FeatureVector(
        candidate_id="TEST_1",
        years_experience=7.6,
        implied_experience_years=7.6,
        experience_gap=0.0,
        total_jobs=2,
        avg_job_duration_months=42,
        max_job_duration_months=60,
        min_job_duration_months=24,
        total_skill_count=15,
        ai_skill_count=9,
        expert_skill_count=5,
        advanced_skill_count=5,
        intermediate_skill_count=0,
        beginner_skill_count=0,
        proficiency_weighted_skill_score=30.0,
        endorsement_weighted_skill_score=1500.0,
        evidence_retrieval=3.0,
        evidence_ranking=3.0,
        evidence_recommendation=0.0,
        evidence_search=2.0,
        evidence_relevance=1.0,
        evidence_personalization=0.0,
        evidence_evaluation=0.0,
        evidence_machine_learning=0.0,
        evidence_nlp=0.0,
        evidence_production_ml=0.0,
        education_count=2,
        highest_education_tier="tier_1",
        tier1_count=1,
        tier2_count=0,
        tier3_count=0,
        tier4_count=0,
        cs_degree_flag=1,
        ai_degree_flag=1,
        recruiter_response_rate=1.0,
        interview_completion_rate=1.0,
        offer_acceptance_rate=1.0,
        profile_views_30d=100,
        applications_30d=0,
        search_appearance_30d=200,
        saved_by_recruiters_30d=10,
        endorsements_received=1500,
        open_to_work_flag=1,
        willing_to_relocate=1,
        engagement_score=1.0,
        verified_email=1,
        verified_phone=1,
        linkedin_connected=1,
        github_available=1,
        github_activity_score=80.0,
        assessment_count=2,
        avg_assessment_score=90.0,
        verification_score=1.0,
        notice_period_days=30,
        availability_score=1.0,
        trust_score=1.0,
        honeypot_flag_count=0,
        salary_inverted_flag=0,
        skill_duration_overflow_flag=0,
        date_paradox_flag=0,
        inflated_experience_flag=0,
        unendorsed_expert_flag=0,
    )
    
    score = CandidateScore(
        candidate_id="TEST_1",
        raw_score=0.8,
        normalized_score=0.8,
        score_breakdown={}
    )
    
    # We don't strictly need a fully populated Candidate for our generator right now, 
    # since all rules pull from FeatureVector, but we provide a mock.
    c = Candidate(candidate_id="TEST_1", profile=None, career_history=[], education=[], skills=[], certifications=[], languages=[], redrob_signals=None)
    
    reasoning = generate_reasoning(c, fv, score)
    print("Output:", reasoning)
    
    assert "7.6" in reasoning
    assert "ranking systems" in reasoning or "retrieval" in reasoning or "search" in reasoning
    assert "skills" in reasoning or "credentials" in reasoning or "assessment" in reasoning
    assert "Note:" not in reasoning # Trust is 1.0, notice is 30, skills > 3, verified > 0.4
    assert len(reasoning) <= 200

    # Test concern branch
    fv.trust_score = 0.5
    reasoning2 = generate_reasoning(c, fv, score)
    print("Output 2:", reasoning2)
    assert "Note: inconsistent profile details." in reasoning2

if __name__ == "__main__":
    test_generate_reasoning()
    print("test_generate_reasoning ... PASS")
