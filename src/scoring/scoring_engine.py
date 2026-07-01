"""
src/scoring/scoring_engine.py
=============================
Computes final candidate scores based on the FeatureVector.

Design principles
-----------------
- No ML models or pre-computation required; pure arithmetic scoring.
- Components are scaled to 0-100 for interpretability.
- Final score is a weighted sum of components, scaled to 0-100.
- Honeypot penalty acts as a final multiplicative decay.
- 'ScoreCard' output preserves the full breakdown for explainability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.scoring.feature_extractor import FeatureVector, TitleCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component Weights (Sum = 1.0)
# ---------------------------------------------------------------------------

EVIDENCE_WEIGHT        = 0.25   # Outweighs skills (production experience)
CAREER_FIT_WEIGHT      = 0.20   # JD target constraints (YOE, role types)
AVAILABILITY_WEIGHT    = 0.15   # Notice period is critical
TECHNICAL_FIT_WEIGHT   = 0.15   # Claimed skills (weighted by relevance)
COMPANY_QUALITY_WEIGHT = 0.10   # Product company experience
BEHAVIORAL_WEIGHT      = 0.10   # Response rates, open to work
SKILL_QUALITY_WEIGHT   = 0.05   # Proficiency levels, assessments

assert sum([
    EVIDENCE_WEIGHT, CAREER_FIT_WEIGHT, AVAILABILITY_WEIGHT, 
    TECHNICAL_FIT_WEIGHT, COMPANY_QUALITY_WEIGHT, BEHAVIORAL_WEIGHT, 
    SKILL_QUALITY_WEIGHT
]) == 1.0, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# ScoreCard Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoreCard:
    """
    Detailed scoring output for a single candidate.
    
    All base scores are 0-100.
    `final_score` is 0-100.
    `honeypot_penalty` is 0.0-1.0 (multiplicative decay applied to final score).
    `trust_score` is 0.70-1.00 (soft authenticity multiplier, applied after honeypot).
    `verification_score` is 0-100 (platform-verified signal composite).
    """
    candidate_id: str

    technical_fit_score:   float
    career_fit_score:      float
    behavioral_score:      float
    availability_score:    float
    evidence_score:        float
    company_quality_score: float
    skill_quality_score:   float
    
    honeypot_penalty:      float
    trust_score:           float   = 1.0   # 0.70–1.00 soft authenticity multiplier
    verification_score:    float   = 0.0   # 0–100 platform-verified signal composite
    final_score:           float   = 0.0
    
    score_breakdown:       dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------------

class ScoringEngine:
    """
    Evaluates FeatureVectors to produce ScoreCards.
    Stateless and safe to use across the candidate stream.
    """

    def score(self, fv: FeatureVector) -> ScoreCard:
        """Score a single candidate."""
        
        # 1. Component calculations (0-100)
        technical_fit   = self._calc_technical_fit(fv)
        career_fit      = self._calc_career_fit(fv)
        behavioral      = self._calc_behavioral(fv)
        availability    = self._calc_availability(fv)
        evidence        = self._calc_evidence(fv)
        company_quality = self._calc_company_quality(fv)
        skill_quality   = self._calc_skill_quality(fv)
        
        # 2. Weighted combination (0-100)
        raw_final = (
            technical_fit   * TECHNICAL_FIT_WEIGHT +
            career_fit      * CAREER_FIT_WEIGHT +
            behavioral      * BEHAVIORAL_WEIGHT +
            availability    * AVAILABILITY_WEIGHT +
            evidence        * EVIDENCE_WEIGHT +
            company_quality * COMPANY_QUALITY_WEIGHT +
            skill_quality   * SKILL_QUALITY_WEIGHT
        )
        
        # 3. Honeypot Penalty
        # If confidence is very high (>= 0.85), penalty is 1.0 (zeroes out score).
        # Otherwise, scale penalty by confidence.
        if fv.honeypot_confidence >= 0.85:
            honeypot_penalty = 1.0
        else:
            # e.g., 0.50 confidence -> 50% penalty
            honeypot_penalty = min(fv.honeypot_confidence, 1.0)
            
        final_score = raw_final * (1.0 - honeypot_penalty)
        
        # 4. Trust Multiplier (0.70–1.00)
        # Applied after honeypot decay. Penalises borderline inconsistencies
        # (salary vs YOE mismatch, mild skill inflation, sparse profiles).
        # Rewards well-maintained profiles (trust=1.02 capped at 1.0).
        final_score = final_score * fv.trust_score
        
        # 5. Breakdown tracking
        breakdown = {
            "technical_fit":     round(technical_fit * TECHNICAL_FIT_WEIGHT, 2),
            "career_fit":        round(career_fit * CAREER_FIT_WEIGHT, 2),
            "behavioral":        round(behavioral * BEHAVIORAL_WEIGHT, 2),
            "availability":      round(availability * AVAILABILITY_WEIGHT, 2),
            "evidence":          round(evidence * EVIDENCE_WEIGHT, 2),
            "company_quality":   round(company_quality * COMPANY_QUALITY_WEIGHT, 2),
            "skill_quality":     round(skill_quality * SKILL_QUALITY_WEIGHT, 2),
            "trust_multiplier":  round(fv.trust_score, 4),
            "verification":      round(fv.verification_score, 2),
        }
        
        # 5. Pipeline Patches: Hard Penalty Gates
        # Gate 1: Technical Eligibility
        technical_eligibility = (
            fv.retrieval_skill_count +
            fv.ranking_skill_count +
            fv.vector_db_skill_count +
            fv.retrieval_evidence_count +
            fv.ranking_evidence_count
        )
        
        if technical_eligibility < 2:
            penalty = final_score * 0.95
            final_score *= 0.05
            breakdown["technical_eligibility_penalty"] = -round(penalty, 2)
        elif technical_eligibility < 3:
            penalty = final_score * 0.85
            final_score *= 0.15
            breakdown["technical_eligibility_penalty"] = -round(penalty, 2)
            
        # Gate 2: Non-Technical Title Penalty
        non_tech_titles = [
            "business analyst", "hr manager", "marketing manager", 
            "customer support", "content writer", "operations manager", 
            "graphic designer", "accountant", "sales executive"
        ]
        t_lower = fv.current_title.lower()
        is_explicitly_banned = any(banned in t_lower for banned in non_tech_titles)
        
        if fv.title_category == TitleCategory.NON_TECHNICAL.value or is_explicitly_banned:
            # Change 2: Exception now requires very strong ML substance.
            # Rationale: A genuine BA-turned-ML-practitioner who spent 5+ years
            # building retrieval systems at a product company deserves a chance.
            # But this must be extremely rare — raise all three bars significantly.
            # Old: product_ml_experience_years >= 3.0 AND evidence >= 60
            # New: product_ml_experience_years >= 5.0 AND evidence >= 75 AND technical_fit >= 70
            if not (
                fv.product_ml_experience_years >= 5.0
                and evidence >= 75.0
                and technical_fit >= 70.0
            ):
                penalty = final_score * 0.80
                final_score *= 0.20
                breakdown["non_technical_penalty"] = -round(penalty, 2)
                
        # Gate 3: Evidence Floor
        if evidence < 20.0:
            penalty = final_score * 0.75
            final_score *= 0.25
            breakdown["low_evidence_penalty"] = -round(penalty, 2)
        elif evidence < 30.0:
            penalty = final_score * 0.50
            final_score *= 0.50
            breakdown["low_evidence_penalty"] = -round(penalty, 2)

        return ScoreCard(
            candidate_id=fv.candidate_id,
            technical_fit_score=round(technical_fit, 2),
            career_fit_score=round(career_fit, 2),
            behavioral_score=round(behavioral, 2),
            availability_score=round(availability, 2),
            evidence_score=round(evidence, 2),
            company_quality_score=round(company_quality, 2),
            skill_quality_score=round(skill_quality, 2),
            honeypot_penalty=round(honeypot_penalty, 4),
            trust_score=round(fv.trust_score, 4),
            verification_score=round(fv.verification_score, 2),
            final_score=round(final_score, 2),
            score_breakdown=breakdown,
        )

    def score_batch(self, feature_vectors: list[FeatureVector]) -> list[ScoreCard]:
        """Score a list of FeatureVectors."""
        return [self.score(fv) for fv in feature_vectors]

    # ── Internal Calculation Methods (all return 0-100) ───────────────────────

    def _calc_technical_fit(self, fv: FeatureVector) -> float:
        """
        Rewards explicit skill declarations.
        Retrieval and Ranking > Vector DB / Recommendation > Generic LLM / ML.
        """
        # Weighted points per skill type
        points = (
            fv.retrieval_skill_count      * 4.0 +
            fv.ranking_skill_count        * 4.0 +
            fv.vector_db_skill_count      * 3.0 +
            fv.recommendation_skill_count * 2.0 +
            fv.llm_skill_count            * 1.0 +
            fv.ml_skill_count             * 0.5
        )
        # Cap at 25 points = 100%
        return min((points / 25.0) * 100.0, 100.0)

    def _calc_career_fit(self, fv: FeatureVector) -> float:
        """
        Evaluates overall career trajectory against JD constraints.
        Base 50. Modifiers apply.
        """
        score = 50.0
        
        # Sweet spot: 5-9 years (+20)
        if 5.0 <= fv.years_experience <= 9.0:
            score += 20.0
        elif fv.years_experience < 2.0 or fv.years_experience > 15.0:
            score -= 20.0
            
        # Tenure stability penalty (Job hoppers)
        if fv.total_roles > 1 and 0 < fv.avg_tenure_months < 18.0:
            score -= 15.0
            
        # Product ML Experience (Massive boost: up to +35)
        ml_boost = min(fv.product_ml_experience_years * 7.0, 35.0)
        score += ml_boost
        
        # Technical ratio multiplier (e.g., 0.5 ratio -> halving the score above 50)
        # We'll just add up to +10 for fully technical careers
        score += fv.technical_role_ratio * 10.0
        
        return max(0.0, min(score, 100.0))

    def _calc_behavioral(self, fv: FeatureVector) -> float:
        """
        Evaluates responsiveness and openness.
        """
        score = 0.0
        
        if fv.open_to_work:
            score += 30.0
            
        score += (fv.recruiter_response_rate * 30.0)
        score += (fv.interview_completion_rate * 20.0)
        
        if fv.github_activity_score != -1.0:
            # Normalize github activity (typically 0-100)
            gh_points = min((fv.github_activity_score / 100.0) * 20.0, 20.0)
            score += gh_points

        # Change 4: Activity recency penalty.
        # The JD says: "a perfect-on-paper candidate who hasn't logged in for
        # 6 months... is not actually available for hiring purposes."
        if fv.days_since_active > 365:
            score -= 30.0   # >1 year inactive — treat as effectively out of market
        elif fv.days_since_active > 180:
            score -= 15.0   # 6–12 months inactive — cooling off, needs fresh outreach

        return min(max(score, 0.0), 100.0)

    def _calc_availability(self, fv: FeatureVector) -> float:
        """
        Evaluates logistical availability (Notice period, response time).
        Starts at 100, drops with penalties.
        """
        score = 100.0
        
        # Notice Period (Median is 90. JD wants <30)
        if fv.notice_period_days <= 30:
            pass # Ideal
        elif fv.notice_period_days <= 60:
            score -= 10.0
        elif fv.notice_period_days <= 90:
            score -= 30.0
        else:
            score -= 60.0 # Huge penalty for >90
            
        # Response time penalty
        if fv.avg_response_time_hours > 72.0:
            score -= 20.0
            
        return max(0.0, score)

    def _calc_evidence(self, fv: FeatureVector) -> float:
        """
        Evaluates actual career descriptions (action verbs, real usage).
        Outweighs technical fit (which is just skill tags).
        """
        points = (
            fv.retrieval_evidence_count      * 5.0 +
            fv.ranking_evidence_count        * 5.0 +
            fv.recommendation_evidence_count * 3.0 +
            fv.vector_db_evidence_count      * 3.0 +
            fv.ml_production_evidence_count  * 4.0
        )
        # 30 evidence points is enough for 100%
        return min((points / 30.0) * 100.0, 100.0)

    def _calc_company_quality(self, fv: FeatureVector) -> float:
        """
        Evaluates the ratio and count of product company experience.
        Services companies are NOT penalized, they just don't add to this specific score.
        """
        score = 0.0
        
        # Ratio contributes up to 60 points
        score += fv.product_company_ratio * 60.0
        
        # Absolute count contributes up to 40 points
        score += min(fv.product_company_count * 20.0, 40.0)
        
        return min(score, 100.0)

    def _calc_skill_quality(self, fv: FeatureVector) -> float:
        """
        Evaluates proficiency depth, validated assessments, and platform
        verification signals.

        Components (0-100):
        - Expert skills:         up to 40 pts (10 per expert skill, max 4)
        - Advanced skills:       up to 20 pts (2 per advanced skill, max 10)
        - Duration depth:        up to 20 pts (avg_skill_duration / 36 months * 20)
        - Validated assessments: up to 10 pts (assessment score * 0.10)
        - Verification boost:    up to 10 pts (verification_score * 0.10)
          Rewards candidates with verified email/phone, LinkedIn, GitHub.
        """
        score = 0.0
        
        # High proficiency counts
        score += min(fv.expert_skill_count * 10.0, 40.0)
        score += min(fv.advanced_skill_count * 2.0, 20.0)
        
        # Duration depth
        score += min((fv.avg_skill_duration / 36.0) * 20.0, 20.0)
        
        # Skill assessments validated by Redrob
        if fv.has_skill_assessments:
            score += min((fv.average_skill_assessment / 100.0) * 10.0, 10.0)

        # Verification boost: platform-verified contact, LinkedIn, GitHub linkage
        # max verification_score=100 → max +10 pts here
        score += min(fv.verification_score * 0.10, 10.0)
            
        return min(score, 100.0)
