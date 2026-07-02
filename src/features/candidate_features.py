"""
src/features/candidate_features.py
=====================================
Feature extraction layer — converts a Candidate object into a flat
FeatureVector suitable for downstream ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from configs.feature_config import (
    AI_CORE_SKILLS,
    PROFICIENCY_WEIGHTS,
    CAREER_EVIDENCE_KEYWORDS,
)
from configs.skill_taxonomy import (
    CAPABILITY_GROUPS,
    PIPELINE_STAGES,
    PRODUCTION_ML_KEYWORDS,
    EXPERIENCE_QUALITY_KEYWORDS,
)

if TYPE_CHECKING:
    from src.data.parser import Candidate
    from src.data.validators import ValidationResult


@dataclass
class FeatureVector:
    """
    Flat feature representation for a single candidate.
    """
    candidate_id: str

    # ── Experience Features ───────────────────────────────────────────────────
    years_experience: float
    implied_experience_years: float
    experience_gap: float
    total_jobs: int
    avg_job_duration_months: float
    max_job_duration_months: int
    min_job_duration_months: int

    # ── Skill Features ────────────────────────────────────────────────────────
    total_skill_count: int
    ai_skill_count: int
    expert_skill_count: int
    advanced_skill_count: int
    intermediate_skill_count: int
    beginner_skill_count: int
    proficiency_weighted_skill_score: float
    endorsement_weighted_skill_score: float

    # ── Career Evidence Features ──────────────────────────────────────────────
    evidence_retrieval: float
    evidence_ranking: float
    evidence_recommendation: float
    evidence_search: float
    evidence_relevance: float
    evidence_personalization: float
    evidence_evaluation: float
    evidence_machine_learning: float
    evidence_nlp: float
    evidence_production_ml: float

    # ── Education Features ────────────────────────────────────────────────────
    education_count: int
    highest_education_tier: str
    tier1_count: int
    tier2_count: int
    tier3_count: int
    tier4_count: int
    cs_degree_flag: int
    ai_degree_flag: int

    # ── Behavioral Features ───────────────────────────────────────────────────
    recruiter_response_rate: float
    interview_completion_rate: float
    offer_acceptance_rate: float
    profile_views_30d: int
    applications_30d: int
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    endorsements_received: int
    open_to_work_flag: int
    willing_to_relocate: int
    engagement_score: float

    # ── Verification Features ─────────────────────────────────────────────────
    verified_email: int
    verified_phone: int
    linkedin_connected: int
    github_available: int
    github_activity_score: float
    assessment_count: int
    avg_assessment_score: float
    verification_score: float

    # ── Availability Features ─────────────────────────────────────────────────
    notice_period_days: int
    availability_score: float

    # ── Trust Features ────────────────────────────────────────────────────────
    trust_score: float
    honeypot_flag_count: int
    salary_inverted_flag: int
    skill_duration_overflow_flag: int
    date_paradox_flag: int
    inflated_experience_flag: int
    unendorsed_expert_flag: int

    # ── Advanced Composite/Recruiter Realism Features (Task 2, 3, 4, 5) ──────
    capability_score: float = 0.0
    pipeline_score: float = 0.0
    production_score: float = 0.0
    experience_quality_score: float = 0.0
    confidence_category: str = "Medium Confidence"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for pandas DataFrame."""
        return asdict(self)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _count_evidence(text: str, keywords: frozenset[str]) -> float:
    """Count occurrences of keywords in text. Returns normalized term frequency."""
    if not text:
        return 0.0
    text = text.lower()
    count = 0
    for kw in keywords:
        count += text.count(kw)
    # Simple normalization: log(1 + count) to dampen high frequencies
    return math.log1p(count)


def _extract_evidence_features(candidate: 'Candidate') -> dict[str, float]:
    # Combine text fields
    texts = [
        candidate.profile.headline,
        candidate.profile.summary,
        candidate.profile.current_title,
    ]
    for job in candidate.career_history:
        texts.append(job.title)
        texts.append(job.description)
    
    full_text = " ".join(filter(None, texts))
    
    evidence = {}
    for ev_key, keywords in CAREER_EVIDENCE_KEYWORDS.items():
        evidence[f"evidence_{ev_key}"] = _count_evidence(full_text, keywords)
    return evidence


def extract_features(
    candidate: 'Candidate',
    validation: 'ValidationResult'
) -> FeatureVector:
    """
    Extract a comprehensive FeatureVector from a Candidate and ValidationResult.
    """
    # 1. Experience Features
    jobs = candidate.career_history
    job_durations = [j.duration_months for j in jobs]
    total_jobs = len(jobs)
    
    # 2. Skill Features
    skills = candidate.skills
    ai_skills = [s for s in skills if s.name in AI_CORE_SKILLS]
    
    prof_score = 0.0
    endorse_score = 0.0
    for s in ai_skills:
        w = PROFICIENCY_WEIGHTS.get(s.proficiency, 1.0)
        prof_score += w
        endorse_score += w * min(s.endorsements, 50)  # capped endorsements

    # 3. Career Evidence
    evidence_feats = _extract_evidence_features(candidate)

    # 4. Education Features
    edu = candidate.education
    tier_counts = {"tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 0, "unknown": 0}
    for e in edu:
        tier = e.tier if e.tier in tier_counts else "unknown"
        tier_counts[tier] += 1

    edu_text = " ".join([f"{e.degree} {e.field_of_study}".lower() for e in edu])
    cs_degree = 1 if "computer science" in edu_text or "computer engineering" in edu_text else 0
    ai_degree = 1 if "artificial intelligence" in edu_text or "machine learning" in edu_text or "data science" in edu_text else 0

    # 5. Behavioral Features
    rs = candidate.redrob_signals
    # Formula: 40% response rate + 40% interview completion + 20% profile completeness/100
    engagement_score = (
        0.4 * rs.recruiter_response_rate +
        0.4 * rs.interview_completion_rate +
        0.2 * (rs.profile_completeness_score / 100.0)
    )

    # 6. Verification Features
    avg_assessment = rs.avg_assessment_score if rs.avg_assessment_score is not None else 0.0
    verification_score = (
        (1.0 if rs.verified_email else 0.0) * 0.2 +
        (1.0 if rs.verified_phone else 0.0) * 0.2 +
        (1.0 if rs.linkedin_connected else 0.0) * 0.2 +
        (1.0 if rs.has_github else 0.0) * 0.2 +
        (min(1.0, rs.n_assessments / 3.0)) * 0.2
    )

    # 7. Availability Features
    # Score 1.0 for immediate availability (<= 15 days), decays for longer notice periods
    notice = rs.notice_period_days
    availability_score = max(0.0, 1.0 - max(0, notice - 15) / 90.0)
    if rs.open_to_work_flag:
        availability_score = min(1.0, availability_score * 1.2)
    
    # 8. Trust Features
    flags = validation.honeypot_flags
    flag_types = [f.split(":")[0] for f in flags]

    # ── Composite / Recruiter Realism scoring calculations ──────────────────
    skills_lower = {s.name.lower().strip() for s in skills}
    career_texts = [
        candidate.profile.headline,
        candidate.profile.summary,
        candidate.profile.current_title,
    ]
    for job in candidate.career_history:
        career_texts.append(job.title)
        career_texts.append(job.description)
    combined_career_text = " ".join(filter(None, career_texts)).lower()

    # 1. Capability Score (Task 1 & 2)
    matched_groups = set()
    for group_name, keywords in CAPABILITY_GROUPS.items():
        if any(kw in skills_lower or kw in combined_career_text for kw in keywords):
            matched_groups.add(group_name)
    capability_score = len(matched_groups) / len(CAPABILITY_GROUPS) if CAPABILITY_GROUPS else 0.0

    # 2. Pipeline Completeness Score (Task 3)
    # Check what fraction of PIPELINE_STAGES are covered by matched capability groups
    active_stages = [1.0 if stage in matched_groups else 0.0 for stage in PIPELINE_STAGES]
    pipeline_score = sum(active_stages) / len(PIPELINE_STAGES) if PIPELINE_STAGES else 0.0

    # 3. Production ML Score (Task 4)
    matched_prod_categories = 0
    for category, keywords in PRODUCTION_ML_KEYWORDS.items():
        if any(kw in combined_career_text for kw in keywords):
            matched_prod_categories += 1
    production_score = matched_prod_categories / len(PRODUCTION_ML_KEYWORDS) if PRODUCTION_ML_KEYWORDS else 0.0

    # 4. Experience Quality Score (Task 5)
    matched_exp_categories = 0
    for category, keywords in EXPERIENCE_QUALITY_KEYWORDS.items():
        if any(kw in combined_career_text for kw in keywords):
            matched_exp_categories += 1
    experience_quality_score = matched_exp_categories / len(EXPERIENCE_QUALITY_KEYWORDS) if EXPERIENCE_QUALITY_KEYWORDS else 0.0

    # 5. Confidence Category
    if validation.trust_score < 0.8 or validation.n_flags > 1:
        confidence_category = "Needs Additional Verification"
    elif (rs.profile_completeness_score >= 70.0 and 
          capability_score >= 0.4 and 
          rs.recruiter_response_rate >= 0.5 and 
          validation.trust_score >= 0.9):
        confidence_category = "High Confidence"
    else:
        confidence_category = "Medium Confidence"

    fv = FeatureVector(
        candidate_id=candidate.candidate_id,
        
        years_experience=candidate.profile.years_of_experience,
        implied_experience_years=candidate.implied_experience_years,
        experience_gap=candidate.experience_discrepancy,
        total_jobs=total_jobs,
        avg_job_duration_months=_safe_divide(sum(job_durations), total_jobs),
        max_job_duration_months=max(job_durations) if job_durations else 0,
        min_job_duration_months=min(job_durations) if job_durations else 0,
        
        total_skill_count=candidate.n_skills,
        ai_skill_count=len(ai_skills),
        expert_skill_count=sum(1 for s in skills if s.proficiency == "expert"),
        advanced_skill_count=sum(1 for s in skills if s.proficiency == "advanced"),
        intermediate_skill_count=sum(1 for s in skills if s.proficiency == "intermediate"),
        beginner_skill_count=sum(1 for s in skills if s.proficiency == "beginner"),
        proficiency_weighted_skill_score=prof_score,
        endorsement_weighted_skill_score=endorse_score,
        
        **evidence_feats,
        
        education_count=len(edu),
        highest_education_tier=candidate.highest_edu_tier,
        tier1_count=tier_counts["tier_1"],
        tier2_count=tier_counts["tier_2"],
        tier3_count=tier_counts["tier_3"],
        tier4_count=tier_counts["tier_4"],
        cs_degree_flag=cs_degree,
        ai_degree_flag=ai_degree,
        
        recruiter_response_rate=rs.recruiter_response_rate,
        interview_completion_rate=rs.interview_completion_rate,
        offer_acceptance_rate=rs.offer_acceptance_rate,
        profile_views_30d=rs.profile_views_received_30d,
        applications_30d=rs.applications_submitted_30d,
        search_appearance_30d=rs.search_appearance_30d,
        saved_by_recruiters_30d=rs.saved_by_recruiters_30d,
        endorsements_received=rs.endorsements_received,
        open_to_work_flag=1 if rs.open_to_work_flag else 0,
        willing_to_relocate=1 if rs.willing_to_relocate else 0,
        engagement_score=engagement_score,
        
        verified_email=1 if rs.verified_email else 0,
        verified_phone=1 if rs.verified_phone else 0,
        linkedin_connected=1 if rs.linkedin_connected else 0,
        github_available=1 if rs.has_github else 0,
        github_activity_score=rs.github_activity_score if rs.has_github else 0.0,
        assessment_count=rs.n_assessments,
        avg_assessment_score=avg_assessment,
        verification_score=verification_score,
        
        notice_period_days=rs.notice_period_days,
        availability_score=availability_score,
        
        trust_score=validation.trust_score,
        honeypot_flag_count=validation.n_flags,
        salary_inverted_flag=1 if "HP-01_SALARY_INVERTED" in flag_types else 0,
        skill_duration_overflow_flag=1 if "HP-02_SKILL_DURATION_OVERFLOW" in flag_types else 0,
        date_paradox_flag=1 if "HP-03_DATE_PARADOX" in flag_types else 0,
        inflated_experience_flag=1 if "HP-04_INFLATED_YOE" in flag_types else 0,
        unendorsed_expert_flag=1 if "HP-05_UNENDORSED_EXPERT" in flag_types else 0,

        capability_score=capability_score,
        pipeline_score=pipeline_score,
        production_score=production_score,
        experience_quality_score=experience_quality_score,
        confidence_category=confidence_category,
    )
    return fv
