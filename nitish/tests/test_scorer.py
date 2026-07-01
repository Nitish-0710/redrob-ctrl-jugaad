"""
tests/test_scorer.py
====================
Unit tests for the ranking scorer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.candidate_features import FeatureVector
from src.ranking.jd_parser import parse_jd
from src.ranking.scorer import score_candidate, rank_candidates


def test_scorer_basic():
    # 1. Get JD
    jd = parse_jd()
    
    # 2. Mock a perfect candidate feature vector
    fv_perfect = FeatureVector(
        candidate_id="PERFECT_1",
        years_experience=7.0,
        implied_experience_years=7.0,
        experience_gap=0.0,
        total_jobs=2,
        avg_job_duration_months=42.0,
        max_job_duration_months=60,
        min_job_duration_months=24,
        
        total_skill_count=15,
        ai_skill_count=10,
        expert_skill_count=5,
        advanced_skill_count=5,
        intermediate_skill_count=0,
        beginner_skill_count=0,
        proficiency_weighted_skill_score=30.0,
        endorsement_weighted_skill_score=1500.0,
        
        evidence_retrieval=3.0,
        evidence_ranking=3.0,
        evidence_recommendation=3.0,
        evidence_search=3.0,
        evidence_relevance=3.0,
        evidence_personalization=3.0,
        evidence_evaluation=3.0,
        evidence_machine_learning=3.0,
        evidence_nlp=3.0,
        evidence_production_ml=3.0,
        
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
        github_activity_score=100.0,
        assessment_count=3,
        avg_assessment_score=100.0,
        verification_score=1.0,
        
        notice_period_days=15,
        availability_score=1.0,
        
        trust_score=1.0,
        honeypot_flag_count=0,
        salary_inverted_flag=0,
        skill_duration_overflow_flag=0,
        date_paradox_flag=0,
        inflated_experience_flag=0,
        unendorsed_expert_flag=0,
    )
    
    score = score_candidate(fv_perfect, jd)
    
    assert score.candidate_id == "PERFECT_1"
    # Given all inputs were maxed out or in ideal range, normalized score should be close to 1.0
    assert score.normalized_score > 0.9
    assert score.score_breakdown["trust_score"] == 1.0
    assert score.score_breakdown["domain_score"] == 1.0
    
    # 3. Mock a poor/fraudulent candidate
    fv_poor = FeatureVector(
        candidate_id="POOR_1",
        years_experience=1.0,
        implied_experience_years=1.0,
        experience_gap=0.0,
        total_jobs=1,
        avg_job_duration_months=12.0,
        max_job_duration_months=12,
        min_job_duration_months=12,
        total_skill_count=2,
        ai_skill_count=0,
        expert_skill_count=0,
        advanced_skill_count=0,
        intermediate_skill_count=0,
        beginner_skill_count=0,
        proficiency_weighted_skill_score=0.0,
        endorsement_weighted_skill_score=0.0,
        
        evidence_retrieval=0.0,
        evidence_ranking=0.0,
        evidence_recommendation=0.0,
        evidence_search=0.0,
        evidence_relevance=0.0,
        evidence_personalization=0.0,
        evidence_evaluation=0.0,
        evidence_machine_learning=0.0,
        evidence_nlp=0.0,
        evidence_production_ml=0.0,
        
        education_count=0,
        highest_education_tier="unknown",
        tier1_count=0,
        tier2_count=0,
        tier3_count=0,
        tier4_count=0,
        cs_degree_flag=0,
        ai_degree_flag=0,
        
        recruiter_response_rate=0.0,
        interview_completion_rate=0.0,
        offer_acceptance_rate=0.0,
        profile_views_30d=0,
        applications_30d=0,
        search_appearance_30d=0,
        saved_by_recruiters_30d=0,
        endorsements_received=0,
        open_to_work_flag=0,
        willing_to_relocate=0,
        engagement_score=0.0,
        
        verified_email=0,
        verified_phone=0,
        linkedin_connected=0,
        github_available=0,
        github_activity_score=0.0,
        assessment_count=0,
        avg_assessment_score=0.0,
        verification_score=0.0,
        
        notice_period_days=90,
        availability_score=0.0,
        
        trust_score=0.25, # heavily penalized
        honeypot_flag_count=5,
        salary_inverted_flag=1,
        skill_duration_overflow_flag=1,
        date_paradox_flag=1,
        inflated_experience_flag=1,
        unendorsed_expert_flag=1,
    )
    
    score_poor = score_candidate(fv_poor, jd)
    assert score_poor.normalized_score < 0.1
    
    # 4. Test ranking
    ranked = rank_candidates([fv_poor, fv_perfect], jd)
    assert ranked[0].candidate_id == "PERFECT_1"
    assert ranked[1].candidate_id == "POOR_1"


if __name__ == "__main__":
    test_scorer_basic()
    print("test_scorer_basic ... PASS")
    print("\nAll scorer tests PASSED.")
