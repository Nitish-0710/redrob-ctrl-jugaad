"""
src/ranking/scorer.py
=====================
Recruiter-style heuristic ranking engine.

Computes interpretable sub-scores across 8 dimensions for each candidate,
then combines them into a normalized final score using config weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

from configs.ranking_config import COMPONENT_WEIGHTS
from src.features.candidate_features import FeatureVector
from src.ranking.jd_parser import JobRequirements
from src.utils.misc import normalize, clamp


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateScore:
    """The final scored result for a candidate."""
    candidate_id: str
    raw_score: float             # Before trust multiplier
    normalized_score: float      # Final score (raw * trust)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    confidence_category: str = "Medium Confidence"


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _score_domain(fv: FeatureVector, jd: JobRequirements) -> float:
    """Match candidate domain evidence against JD required domains."""
    score = 0.0
    max_possible = 0.0
    
    # 3.0 is a reasonable empirical max for log1p keyword counts
    MAX_EVIDENCE_CAP = 3.0 
    
    domains = [
        (jd.domains.retrieval, fv.evidence_retrieval),
        (jd.domains.ranking, fv.evidence_ranking),
        (jd.domains.recommendation, fv.evidence_recommendation),
        (jd.domains.search, fv.evidence_search),
        (jd.domains.relevance, fv.evidence_relevance),
        (jd.domains.personalization, fv.evidence_personalization),
        (jd.domains.evaluation, fv.evidence_evaluation),
        (jd.domains.machine_learning, fv.evidence_machine_learning),
        (jd.domains.nlp, fv.evidence_nlp),
        (jd.domains.production_ml, fv.evidence_production_ml),
    ]
    
    for is_required, evidence_val in domains:
        if is_required:
            max_possible += 1.0
            score += normalize(evidence_val, 0.0, MAX_EVIDENCE_CAP)
            
    if max_possible == 0:
        return 0.0
    return score / max_possible


def _score_experience(fv: FeatureVector, jd: JobRequirements) -> float:
    """Score YoE against JD preferred bounds."""
    yoe = fv.implied_experience_years
    
    if yoe < jd.experience.minimum_years:
        return 0.0
        
    pref_min = jd.experience.preferred_min_years
    pref_max = jd.experience.preferred_max_years
    
    if pref_min <= yoe <= pref_max:
        return 1.0
    
    # Decay outside preferred range
    if yoe < pref_min:
        return max(0.0, 1.0 - (pref_min - yoe) * 0.2)
    else:
        # Overqualified: mild penalty
        return max(0.5, 1.0 - (yoe - pref_max) * 0.05)


def _score_skills(fv: FeatureVector, jd: JobRequirements) -> float:
    """Combine proficiency and peer endorsements for AI skills."""
    # Based on 99th percentile distributions from dataset forensics
    MAX_PROF = 30.0
    MAX_END = 1500.0
    
    p_score = normalize(fv.proficiency_weighted_skill_score, 0.0, MAX_PROF)
    e_score = normalize(fv.endorsement_weighted_skill_score, 0.0, MAX_END)
    
    # 70% weight on claimed proficiency, 30% weight on peer endorsements
    return p_score * 0.7 + e_score * 0.3


def _score_education(fv: FeatureVector, jd: JobRequirements) -> float:
    """Score education relevance and prestige."""
    if fv.education_count == 0:
        return 0.0
        
    # Check degree relevance against JD
    rel_score = 0.0
    if fv.cs_degree_flag or fv.ai_degree_flag:
        rel_score = 1.0
    else:
        rel_score = 0.5  # Has some degree, but maybe not CS/AI
        
    # Tier prestige
    tier_score = 0.0
    if fv.tier1_count > 0: tier_score = 1.0
    elif fv.tier2_count > 0: tier_score = 0.8
    elif fv.tier3_count > 0: tier_score = 0.6
    elif fv.tier4_count > 0: tier_score = 0.4
    else: tier_score = 0.2
    
    return rel_score * 0.6 + tier_score * 0.4


def _score_behavioral(fv: FeatureVector) -> float:
    """Score candidate engagement and responsiveness."""
    return fv.engagement_score


def _score_verification(fv: FeatureVector) -> float:
    """Score verified signals (GitHub, assessments, contact info)."""
    return fv.verification_score


def _score_availability(fv: FeatureVector) -> float:
    """Score notice period and willingness to work."""
    return fv.availability_score


def _score_trust(fv: FeatureVector) -> float:
    """Return the trust multiplier derived from honeypot validators."""
    return fv.trust_score


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCORING INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def score_candidate(fv: FeatureVector, jd: JobRequirements) -> CandidateScore:
    """
    Compute the complete scoring breakdown for a single candidate against a JD.
    """
    # Compute base original scores
    orig_domain = _score_domain(fv, jd)
    orig_experience = _score_experience(fv, jd)
    orig_skills = _score_skills(fv, jd)
    orig_education = _score_education(fv, jd)
    orig_behavioral = _score_behavioral(fv)
    orig_verification = _score_verification(fv)
    orig_availability = _score_availability(fv)

    # Blend new composite features (Tasks 2, 3, 4, 5)
    # Blend capability_score and pipeline_score into skill_score (original weight: 0.25)
    if fv.capability_score == 0.0 and fv.pipeline_score == 0.0:
        skill_score = orig_skills
    else:
        skill_score = 0.5 * orig_skills + 0.3 * fv.capability_score + 0.2 * fv.pipeline_score

    # Blend experience_quality_score and production_score into experience_score (original weight: 0.15)
    if fv.experience_quality_score == 0.0 and fv.production_score == 0.0:
        experience_score = orig_experience
    else:
        experience_score = 0.5 * orig_experience + 0.3 * fv.experience_quality_score + 0.2 * fv.production_score

    components = {
        "domain_score":       orig_domain,
        "experience_score":   experience_score,
        "skill_score":        skill_score,
        "education_score":    orig_education,
        "behavioral_score":   orig_behavioral,
        "verification_score": orig_verification,
        "availability_score": orig_availability,
    }
    
    # Trust is handled as a final multiplier
    trust_multiplier = _score_trust(fv)
    
    # Calculate base score using config weights
    raw_score = 0.0
    for name, score in components.items():
        weight = COMPONENT_WEIGHTS.get(name, 0.0)
        raw_score += score * weight
        
    final_score = raw_score * trust_multiplier
    
    # Include trust in the breakdown for transparency
    components["trust_score"] = trust_multiplier
    
    # Add composite components to breakdown for debugging/explainability if needed
    components["raw_capability_score"] = fv.capability_score
    components["raw_pipeline_score"] = fv.pipeline_score
    components["raw_production_score"] = fv.production_score
    components["raw_experience_quality_score"] = fv.experience_quality_score
    
    return CandidateScore(
        candidate_id=fv.candidate_id,
        raw_score=raw_score,
        normalized_score=final_score,
        score_breakdown=components,
        confidence_category=fv.confidence_category
    )


def rank_candidates(fvs: List[FeatureVector], jd: JobRequirements) -> List[CandidateScore]:
    """
    Score a batch of candidates and return them sorted by highest score.
    """
    scored = [score_candidate(fv, jd) for fv in fvs]
    scored.sort(key=lambda x: (-x.normalized_score, x.candidate_id))
    return scored
