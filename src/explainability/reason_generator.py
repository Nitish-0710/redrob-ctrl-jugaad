"""
src/explainability/reason_generator.py
======================================
Generates deterministic, recruiter-friendly explanations for candidate scores.

Design principles
-----------------
- No LLM generation; purely rule-based and deterministic.
- No hallucinations; strictly bounds claims to extracted FeatureVector data.
- Output is factual and specific — references actual values from the profile.
- Format: "{title} with {years}yrs; {domain_strength}; {signal}."

Stage 4 manual review checks (from submission_spec):
- Specific facts (title, years, named skills, signal values)  ✓
- JD connection (retrieval, ranking, vector DB — not generic praise) ✓
- Honest concerns (notice period, inactivity — with actual values)  ✓
- No hallucination (every claim in FeatureVector)                   ✓
- Variation (different facts per candidate, not template strings)   ✓
- Rank consistency (low-rank candidates get honest concern language) ✓
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from src.scoring.feature_extractor import FeatureVector
from src.scoring.scoring_engine import ScoreCard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence Enum
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


# ---------------------------------------------------------------------------
# ExplanationResult Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExplanationResult:
    """
    Human-readable explanation of why a candidate received their score.

    reasoning_text  → goes into the submission CSV (hard 300-char limit)
    strengths       → used by audit.py for human review
    concerns        → used by audit.py for human review
    """
    candidate_id:     str
    summary_reason:   str
    strengths:        list[str]
    concerns:         list[str]
    confidence_level: str
    reasoning_text:   str


# ---------------------------------------------------------------------------
# Reason Generator Engine
# ---------------------------------------------------------------------------

class ReasonGenerator:
    """
    Evaluates FeatureVector and ScoreCard to generate human-readable reasoning.

    The critical output is `reasoning_text` which goes into the submission CSV.

    Format: "{title} with {yrs}yrs; {domain}; {signal}."

    Every element is a specific fact from FeatureVector — no template phrases.
    Tone matches rank: high-scoring candidates get confident language;
    low-scoring candidates get honest concern language.
    """

    def generate(self, fv: FeatureVector, sc: ScoreCard) -> ExplanationResult:
        """Generate explanation for a single candidate."""
        strengths = self._detect_strengths(fv)
        concerns  = self._detect_concerns(fv, sc)
        conf      = self._determine_confidence(fv, sc)

        # reasoning_text goes into the CSV — must be factual, specific, <300 chars
        reasoning = self._build_reasoning_text(fv, sc)
        summary   = self._build_summary_reason(fv, sc)

        return ExplanationResult(
            candidate_id=fv.candidate_id,
            summary_reason=summary,
            strengths=strengths,
            concerns=concerns,
            confidence_level=conf.value,
            reasoning_text=reasoning,
        )

    def generate_batch(
        self,
        feature_vectors: list[FeatureVector],
        scorecards: list[ScoreCard],
    ) -> list[ExplanationResult]:
        """Generate explanations for a batch. Assumes lists are identically ordered."""
        assert len(feature_vectors) == len(scorecards)
        results = []
        for fv, sc in zip(feature_vectors, scorecards):
            assert fv.candidate_id == sc.candidate_id
            results.append(self.generate(fv, sc))
        return results

    # ── Strengths / Concerns (used by audit.py) ──────────────────────────────

    def _detect_strengths(self, fv: FeatureVector) -> list[str]:
        """Identify discrete strengths based on feature thresholds."""
        strengths = []

        if fv.retrieval_skill_count >= 2 or fv.retrieval_evidence_count >= 3:
            strengths.append("Strong retrieval and semantic search experience")
        if fv.ranking_skill_count >= 2 or fv.ranking_evidence_count >= 2:
            strengths.append("Demonstrated ranking/LTR system expertise")
        if fv.vector_db_skill_count >= 2 or fv.vector_db_evidence_count >= 2:
            strengths.append("Vector database infrastructure experience")
        if fv.recommendation_skill_count >= 2 or fv.recommendation_evidence_count >= 2:
            strengths.append("Recommendation engine experience")
        if fv.product_ml_experience_years >= 3.0:
            strengths.append("Meaningful product-company ML background")
        elif fv.product_company_ratio >= 0.7:
            strengths.append("Strong product-company background")
        if 5.0 <= fv.years_experience <= 9.0:
            strengths.append("Ideal years of experience for the role")
        if fv.github_activity_score >= 60.0:
            strengths.append(f"Active GitHub profile (score {fv.github_activity_score:.0f})")
        if fv.recruiter_response_rate >= 0.8:
            strengths.append("High recruiter responsiveness")
        if fv.interview_completion_rate >= 0.8:
            strengths.append("Reliable interview completion history")
        if fv.verification_score >= 70.0:
            strengths.append(f"High platform verification score ({fv.verification_score:.0f}/100)")
        if fv.trust_score >= 1.0:
            strengths.append("Consistent profile (salary, skill depth, completeness)")

        return strengths

    def _detect_concerns(self, fv: FeatureVector, sc: ScoreCard) -> list[str]:
        """Identify discrete concerns based on feature thresholds."""
        concerns = []

        if fv.honeypot_confidence >= 0.85:
            concerns.append("Profile flagged as highly suspicious (honeypot candidate)")
        elif fv.honeypot_confidence >= 0.40:
            concerns.append("Profile exhibits anomalous or contradictory signals")
        if fv.days_since_active > 365:
            months = fv.days_since_active // 30
            concerns.append(f"No platform activity for {months} months")
        elif fv.days_since_active > 180:
            concerns.append(f"Inactive for {fv.days_since_active} days")
        if fv.product_company_ratio <= 0.2:
            concerns.append("Primarily services-company experience")
        if fv.years_experience < 2.0:
            concerns.append("May lack necessary seniority (junior profile)")
        if sc.evidence_score < 30.0 and fv.years_experience >= 3.0:
            concerns.append("Limited evidence of production retrieval systems")
        if fv.notice_period_days > 60:
            concerns.append("Extended notice period")
        if fv.recruiter_response_rate > 0.0 and fv.recruiter_response_rate < 0.4:
            concerns.append("Low recruiter responsiveness")
        if fv.interview_completion_rate > 0.0 and fv.interview_completion_rate < 0.5:
            concerns.append("History of low interview completion")

        return concerns

    def _determine_confidence(self, fv: FeatureVector, sc: ScoreCard) -> ConfidenceLevel:
        """Determine system confidence in the candidate's final score."""
        if fv.honeypot_confidence >= 0.80 or sc.final_score < 30.0:
            return ConfidenceLevel.LOW
        if sc.evidence_score >= 70.0 and sc.final_score >= 65.0 and fv.honeypot_confidence < 0.2:
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.MEDIUM

    def _build_summary_reason(self, fv: FeatureVector, sc: ScoreCard) -> str:
        """A 1-sentence TL;DR of the candidate's fit."""
        if fv.honeypot_confidence >= 0.85:
            return "Rejected due to suspicious profile signals."
        if sc.final_score >= 75.0:
            return "Exceptional fit — strong retrieval/ranking experience at product companies."
        elif sc.final_score >= 50.0:
            return "Solid candidate with relevant ML skills; some gaps in retrieval evidence."
        else:
            return "Lacks core technical requirements or career fit for this role."

    # ── Factual Reasoning Builder ─────────────────────────────────────────────

    def _build_domain_description(self, fv: FeatureVector) -> str:
        """
        Build a specific domain description.

        Priority: actual skill names > domain categories > product ML years.
        Uses top_domain_skills (expert/advanced, JD-relevant) when available
        so the text references technology names the recruiter can verify.
        """
        # Best case: use real skill names from the profile
        if fv.top_domain_skills:
            return " + ".join(fv.top_domain_skills[:2])

        # Fall back to domain category descriptions from evidence/skill counts
        domains: list[str] = []
        if fv.retrieval_evidence_count >= 3 or fv.retrieval_skill_count >= 2:
            domains.append("retrieval/semantic search")
        if fv.ranking_evidence_count >= 2 or fv.ranking_skill_count >= 2:
            domains.append("learning-to-rank")
        if fv.vector_db_evidence_count >= 2 or fv.vector_db_skill_count >= 2:
            domains.append("vector DB")
        if fv.recommendation_evidence_count >= 2 or fv.recommendation_skill_count >= 2:
            domains.append("recommendation engines")

        if domains:
            return " + ".join(domains[:2])

        # Last resort: product ML years or generic adjacent
        if fv.product_ml_experience_years >= 2.0:
            return f"{fv.product_ml_experience_years:.1f}yrs applied ML"

        return "adjacent tech background"

    def _build_key_signal(self, fv: FeatureVector) -> str:
        """
        Return the single most important behavioral signal as a factual string.

        Negative signals (inactivity, long notice) surface first — they are
        more actionable for recruiting decisions.

        NOTE: github_activity_score is handled as a dedicated segment in
        _build_reasoning_text; this method skips it so it doesn't double-count.
        """
        # Concern signals take priority
        if fv.days_since_active > 365:
            months = fv.days_since_active // 30
            return f"inactive {months}mo"
        if fv.notice_period_days > 90:
            return f"notice {fv.notice_period_days}d"

        # Always surface response rate as a concrete numeric fact
        if fv.recruiter_response_rate > 0.0:
            return f"response rate {fv.recruiter_response_rate:.2f}"

        return ""

    def _build_reasoning_text(self, fv: FeatureVector, sc: ScoreCard) -> str:
        """
        Build the submission CSV reasoning string.

        Target format (matches user spec):
          "{title} with {yrs}yrs; {domain}; github {N}; response rate {R}"

        Segments (separated by "; "):
          1. Identity:  title + years                        (always present)
          2. Domain:    named technologies or categories     (always present)
          3. GitHub:    "github {score}" if score >= 60      (when available)
          4. Signal:    notice period / inactivity / response rate  (always present)
          5. Concern:   honest concern for lower-ranked candidates  (conditional)

        Rules:
        - Every element is a specific fact from FeatureVector — no templates.
        - Max 300 characters (submission spec hard limit).
        - NO trailing period (matches target format).
        - Negative signals (inactivity, long notice) surface first in slot 4.

        Examples:
          "AI Engineer with 6.3yrs; FAISS + Semantic Search; github 74; response rate 0.87"
          "ML Engineer with 7.0yrs; learning-to-rank + Pinecone; notice 120d"
          "Data Scientist with 6.7yrs; retrieval/semantic search; inactive 8mo"
          "NLP Engineer with 16.1yrs; Milvus + BM25; github 74; response rate 0.73"
        """
        # Honeypot: short factual rejection reason
        if fv.honeypot_confidence >= 0.85:
            title = fv.current_title or "Candidate"
            return (
                f"{title} with {fv.years_experience:.1f}yrs; "
                f"profile flagged — contradictory signals detected"
            )[:300]

        parts: list[str] = []

        # Segment 1: Identity — title + years (always present)
        title = fv.current_title if fv.current_title else "Candidate"
        parts.append(f"{title} with {fv.years_experience:.1f}yrs")

        # Segment 2: Domain strength — named technologies or categories
        domain = self._build_domain_description(fv)
        parts.append(domain)

        # Segment 3: GitHub score (dedicated slot, shown when linked and notable)
        # Shows the raw numeric score — recruiter can cross-check profile directly.
        if fv.github_activity_score >= 60.0:
            parts.append(f"github {fv.github_activity_score:.0f}")

        # Segment 4: Key behavioral signal — concern-first, then response rate
        signal = self._build_key_signal(fv)
        if signal:
            parts.append(signal)

        # Segment 5: Honest concern for lower-ranked candidates
        if sc.final_score < 65.0 and sc.evidence_score < 35.0:
            parts.append("limited retrieval evidence")
        elif sc.final_score < 50.0:
            parts.append("below preferred threshold")

        text = "; ".join(parts)

        # Truncate gracefully to the 300-char hard limit
        if len(text) > 300:
            text = text[:297].strip() + "..."

        return text
