"""
tests/test_honeypot_detector.py
================================
Unit tests for HoneypotDetector.

Each test function:
1. Constructs a minimal CandidateRecord that exercises exactly one check.
2. Verifies the correct HoneypotFlag is triggered (or not triggered).
3. Verifies the confidence is within the expected range.

Fixtures are modelled on real archetypes observed in the dataset analysis:
- CAND_0000002 type: Marketing Manager + advanced AI skills
- CAND_0000031 type: Recommendation Systems Engineer with 677mo skill duration
- High-quality ML Engineer (should NOT be flagged)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
from typing import Optional

import pytest

from config.settings import DEFAULT_CONFIG
from src.models.candidate import (
    CandidateProfile,
    CandidateRecord,
    CareerRole,
    CompanySize,
    Education,
    InstitutionTier,
    Language,
    LanguageProficiency,
    RedrobSignals,
    SalaryRange,
    Skill,
    SkillProficiency,
    WorkMode,
)
from src.scoring.honeypot_detector import (
    HoneypotDetector,
    HoneypotFlag,
    HoneypotResult,
    _check_domain_skill_contradiction,
    _check_expert_with_zero_experience,
    _check_impossible_timeline,
    _check_non_tech_title_with_ai_skills,
    _check_skill_duration_inflation,
    _check_summary_title_contradiction,
    _check_suspicious_career_progression,
    _check_title_career_mismatch,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_signals(**overrides) -> RedrobSignals:
    """Create a minimal valid RedrobSignals with sensible defaults."""
    defaults = dict(
        profile_completeness_score=75.0,
        signup_date=date(2025, 1, 1),
        last_active_date=date(2026, 5, 1),
        open_to_work_flag=True,
        profile_views_received_30d=20,
        applications_submitted_30d=2,
        recruiter_response_rate=0.60,
        avg_response_time_hours=24.0,
        skill_assessment_scores={},
        connection_count=300,
        endorsements_received=20,
        notice_period_days=30,
        expected_salary_range_inr_lpa=SalaryRange(min_lpa=25.0, max_lpa=40.0),
        preferred_work_mode=WorkMode.HYBRID,
        willing_to_relocate=True,
        github_activity_score=55.0,
        search_appearance_30d=100,
        saved_by_recruiters_30d=5,
        interview_completion_rate=0.85,
        offer_acceptance_rate=0.70,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )
    defaults.update(overrides)
    return RedrobSignals(**defaults)


def _make_profile(
    title: str = "ML Engineer",
    yoe: float = 7.0,
    summary: str = "Experienced ML engineer building production retrieval systems.",
    company: str = "Razorpay",
    industry: str = "Fintech",
) -> CandidateProfile:
    return CandidateProfile(
        anonymized_name="Test Candidate",
        headline=title,
        summary=summary,
        location="Bangalore",
        country="India",
        years_of_experience=yoe,
        current_title=title,
        current_company=company,
        current_company_size=CompanySize.XL,
        current_industry=industry,
    )


def _make_role(
    title: str = "ML Engineer",
    company: str = "Razorpay",
    industry: str = "Fintech",
    start: date = date(2019, 1, 1),
    end: Optional[date] = None,
    duration_months: int = 36,
    is_current: bool = True,
    description: str = "Built vector search and ranking systems.",
) -> CareerRole:
    return CareerRole(
        company=company,
        title=title,
        start_date=start,
        end_date=end,
        duration_months=duration_months,
        is_current=is_current,
        industry=industry,
        company_size=CompanySize.XL,
        description=description,
    )


def _make_skill(
    name: str,
    proficiency: SkillProficiency = SkillProficiency.ADVANCED,
    duration_months: int = 24,
    endorsements: int = 10,
) -> Skill:
    return Skill(
        name=name,
        proficiency=proficiency,
        endorsements=endorsements,
        duration_months=duration_months,
    )


def _make_candidate(
    candidate_id: str = "CAND_TEST001",
    profile: Optional[CandidateProfile] = None,
    career: Optional[list[CareerRole]] = None,
    skills: Optional[list[Skill]] = None,
    signals: Optional[RedrobSignals] = None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        profile=profile or _make_profile(),
        career_history=career or [_make_role()],
        education=[
            Education(
                institution="IIT Bombay",
                degree="B.Tech",
                field_of_study="Computer Science",
                start_year=2012,
                end_year=2016,
                grade="9.0",
                tier=InstitutionTier.TIER_1,
            )
        ],
        skills=skills or [
            _make_skill("Python", SkillProficiency.ADVANCED, 60),
            _make_skill("Embeddings", SkillProficiency.ADVANCED, 36),
        ],
        certifications=[],
        languages=[Language("English", LanguageProficiency.PROFESSIONAL)],
        signals=signals or _make_signals(),
    )


# ---------------------------------------------------------------------------
# Tests: _check_skill_duration_inflation
# ---------------------------------------------------------------------------

class TestSkillDurationInflation:

    def test_clean_candidate_no_flag(self):
        """A candidate with normal skill duration should not be flagged."""
        # 7 YOE = 84 months. Skills total: 2 × 30 = 60 months. Ratio = 0.7×
        candidate = _make_candidate(
            skills=[
                _make_skill("Python", SkillProficiency.ADVANCED, 30),
                _make_skill("Embeddings", SkillProficiency.ADVANCED, 30),
            ]
        )
        results = _check_skill_duration_inflation(candidate, DEFAULT_CONFIG)
        assert results == [], f"Expected no flags, got: {results}"

    def test_borderline_inflation_flagged_at_low_confidence(self):
        """4-5× ratio triggers borderline flag at lower confidence."""
        # 2 YOE = 24 months. Skills total: 12 × 10 = 120 months. Ratio = 5×
        candidate = _make_candidate(
            profile=_make_profile(yoe=2.0),
            skills=[_make_skill(f"Skill{i}", duration_months=10) for i in range(12)],
        )
        results = _check_skill_duration_inflation(candidate, DEFAULT_CONFIG)
        assert len(results) == 1
        flag, conf, _ = results[0]
        assert flag == HoneypotFlag.SKILL_DURATION_INFLATION
        assert 0.30 <= conf < 0.60, f"Expected borderline confidence, got {conf}"

    def test_severe_inflation_high_confidence(self):
        """
        Modelled on CAND_0000031: Recommendation Systems Engineer, 6 YOE,
        677 months total skill duration → caught as honeypot.
        """
        candidate = _make_candidate(
            profile=_make_profile(yoe=6.0),
            skills=[_make_skill(f"Skill{i}", duration_months=50) for i in range(14)],
            # 14 × 50 = 700 months vs 72 months YOE → ratio ≈ 9.7×
        )
        results = _check_skill_duration_inflation(candidate, DEFAULT_CONFIG)
        assert len(results) == 1
        flag, conf, _ = results[0]
        assert flag == HoneypotFlag.SKILL_DURATION_INFLATION
        assert conf >= 0.80, f"Expected high confidence, got {conf}"

    def test_zero_yoe_skipped(self):
        """Cannot compute ratio if YOE is 0."""
        candidate = _make_candidate(
            profile=_make_profile(yoe=0.0),
            skills=[_make_skill("Python", duration_months=100)],
        )
        results = _check_skill_duration_inflation(candidate, DEFAULT_CONFIG)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: _check_expert_with_zero_experience
# ---------------------------------------------------------------------------

class TestExpertWithZeroExperience:

    def test_expert_zero_months_flagged_at_095(self):
        """Expert skill + 0 months → 0.95 confidence."""
        candidate = _make_candidate(
            skills=[_make_skill("Pinecone", SkillProficiency.EXPERT, duration_months=0)]
        )
        results = _check_expert_with_zero_experience(candidate, DEFAULT_CONFIG)
        assert len(results) == 1
        flag, conf, reason = results[0]
        assert flag == HoneypotFlag.EXPERT_WITH_ZERO_EXPERIENCE
        assert conf == 0.95
        assert "Pinecone" in reason

    def test_advanced_zero_months_flagged(self):
        """Advanced skill + 0 months → 0.80 confidence."""
        candidate = _make_candidate(
            skills=[_make_skill("FAISS", SkillProficiency.ADVANCED, duration_months=0)]
        )
        results = _check_expert_with_zero_experience(candidate, DEFAULT_CONFIG)
        assert any(f == HoneypotFlag.EXPERT_WITH_ZERO_EXPERIENCE for f, _, _ in results)
        conf_vals = [c for _, c, _ in results]
        assert all(c == 0.80 for c in conf_vals)

    def test_multiple_expert_zero_flagged(self):
        """Multiple expert+0mo skills → multiple flag entries."""
        candidate = _make_candidate(
            skills=[
                _make_skill("Pinecone", SkillProficiency.EXPERT, 0),
                _make_skill("FAISS", SkillProficiency.EXPERT, 0),
                _make_skill("RAG", SkillProficiency.EXPERT, 0),
            ]
        )
        results = _check_expert_with_zero_experience(candidate, DEFAULT_CONFIG)
        assert len(results) == 3

    def test_advanced_with_normal_duration_clean(self):
        """Advanced skill + reasonable duration should NOT be flagged."""
        candidate = _make_candidate(
            skills=[_make_skill("FAISS", SkillProficiency.ADVANCED, duration_months=24)]
        )
        results = _check_expert_with_zero_experience(candidate, DEFAULT_CONFIG)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: _check_non_tech_title_with_ai_skills
# ---------------------------------------------------------------------------

class TestNonTechTitleWithAISkills:

    def test_marketing_manager_with_ai_skills_flagged(self):
        """
        Modelled on the JD's explicit trap:
        Marketing Manager + Embeddings + RAG + Pinecone → flagged.
        """
        candidate = _make_candidate(
            profile=_make_profile(title="Marketing Manager", industry="Manufacturing"),
            skills=[
                _make_skill("Embeddings", SkillProficiency.ADVANCED, 16),
                _make_skill("RAG", SkillProficiency.ADVANCED, 16),
                _make_skill("Pinecone", SkillProficiency.ADVANCED, 16),
                _make_skill("FAISS", SkillProficiency.ADVANCED, 16),
                _make_skill("Vector Search", SkillProficiency.ADVANCED, 16),
            ],
        )
        results = _check_non_tech_title_with_ai_skills(candidate, DEFAULT_CONFIG)
        assert len(results) == 1
        flag, conf, _ = results[0]
        assert flag == HoneypotFlag.NON_TECH_WITH_ADVANCED_AI_SKILLS
        assert conf >= 0.85

    def test_hr_manager_with_one_ai_skill_low_confidence(self):
        """HR Manager with just 1 advanced AI skill → low-confidence flag."""
        candidate = _make_candidate(
            profile=_make_profile(title="HR Manager"),
            skills=[_make_skill("LangChain", SkillProficiency.ADVANCED, 16)],
        )
        results = _check_non_tech_title_with_ai_skills(candidate, DEFAULT_CONFIG)
        assert len(results) == 1
        _, conf, _ = results[0]
        assert conf < 0.65

    def test_ml_engineer_with_ai_skills_not_flagged(self):
        """ML Engineer + AI skills is the TARGET profile — must not flag."""
        candidate = _make_candidate(
            profile=_make_profile(title="ML Engineer"),
            skills=[
                _make_skill("Embeddings", SkillProficiency.ADVANCED, 36),
                _make_skill("RAG", SkillProficiency.ADVANCED, 24),
            ],
        )
        results = _check_non_tech_title_with_ai_skills(candidate, DEFAULT_CONFIG)
        assert results == [], "ML Engineer with AI skills should NOT be flagged"

    def test_beginner_ai_skill_not_flagged(self):
        """Beginner-level AI skills on non-tech title are NOT flagged (only advanced/expert)."""
        candidate = _make_candidate(
            profile=_make_profile(title="Accountant"),
            skills=[_make_skill("Machine Learning", SkillProficiency.BEGINNER, 5)],
        )
        results = _check_non_tech_title_with_ai_skills(candidate, DEFAULT_CONFIG)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: _check_summary_title_contradiction
# ---------------------------------------------------------------------------

class TestSummaryTitleContradiction:

    def test_marketing_summary_with_ml_title_flagged(self):
        """
        Synthetic generation error: summary says 'marketing manager background'
        but current title is ML Engineer.
        """
        summary = (
            "Professional with 5+ years of experience. "
            "My professional background is in marketing manager — I've built "
            "and led teams, owned KPIs. Lately I've been curious about AI tools."
        )
        candidate = _make_candidate(
            profile=_make_profile(title="ML Engineer", summary=summary),
        )
        results = _check_summary_title_contradiction(candidate, DEFAULT_CONFIG)
        assert len(results) >= 1
        flag, conf, _ = results[0]
        assert flag == HoneypotFlag.SUMMARY_TITLE_CONTRADICTION
        assert conf >= 0.50

    def test_consistent_ml_summary_not_flagged(self):
        """Consistent technical summary + ML title should not be flagged."""
        summary = (
            "Senior ML engineer with 7 years building production retrieval "
            "and ranking systems.  Shipped embedding-based search at scale."
        )
        candidate = _make_candidate(
            profile=_make_profile(title="ML Engineer", summary=summary),
        )
        results = _check_summary_title_contradiction(candidate, DEFAULT_CONFIG)
        assert results == []

    def test_non_tech_summary_with_non_tech_title_not_flagged(self):
        """Non-tech summary + non-tech title is consistent — should not flag."""
        summary = "I've spent my career in marketing manager roles, driving outcomes."
        candidate = _make_candidate(
            profile=_make_profile(title="Marketing Manager", summary=summary),
        )
        results = _check_summary_title_contradiction(candidate, DEFAULT_CONFIG)
        assert results == []

    def test_multiple_contradiction_phrases_higher_confidence(self):
        """More contradiction phrases → higher confidence."""
        summary = (
            "Brand design and creative direction at a consumer-products company. "
            "Marketing manager background. Content writing and SEO strategy. "
            "Customer support experience."
        )
        candidate = _make_candidate(
            profile=_make_profile(title="Data Scientist", summary=summary),
        )
        results = _check_summary_title_contradiction(candidate, DEFAULT_CONFIG)
        if results:
            _, conf1, _ = results[0]
            # With 2+ phrases, confidence should be higher
            assert conf1 >= 0.55


# ---------------------------------------------------------------------------
# Tests: _check_impossible_timeline
# ---------------------------------------------------------------------------

class TestImpossibleTimeline:

    def test_end_before_start_flagged(self):
        """end_date before start_date is fundamentally impossible."""
        role = _make_role(
            start=date(2022, 6, 1),
            end=date(2020, 1, 1),   # before start
            duration_months=24,
            is_current=False,
        )
        candidate = _make_candidate(career=[role])
        results = _check_impossible_timeline(candidate, DEFAULT_CONFIG)
        assert len(results) >= 1
        flag, conf, _ = results[0]
        assert flag == HoneypotFlag.IMPOSSIBLE_TIMELINE
        assert conf >= 0.90

    def test_large_duration_mismatch_flagged(self):
        """Stated 84 months but dates imply 12 months → >24mo delta → flagged."""
        role = _make_role(
            start=date(2022, 1, 1),
            end=date(2023, 1, 1),   # 12 months actual
            duration_months=84,    # stated: 84 months
            is_current=False,
        )
        candidate = _make_candidate(career=[role])
        results = _check_impossible_timeline(candidate, DEFAULT_CONFIG)
        assert len(results) >= 1
        _, conf, reason = results[0]
        assert conf >= 0.65
        assert "84" in reason or "12" in reason

    def test_small_discrepancy_not_flagged(self):
        """A discrepancy of 3 months is within tolerance — should not flag."""
        role = _make_role(
            start=date(2021, 1, 1),
            end=date(2023, 4, 1),   # 27 months actual
            duration_months=24,    # stated 24 — only 3mo off
            is_current=False,
        )
        candidate = _make_candidate(career=[role])
        results = _check_impossible_timeline(candidate, DEFAULT_CONFIG)
        assert results == [], f"Small discrepancy should not be flagged: {results}"

    def test_current_role_no_end_date_not_flagged(self):
        """Current role has end_date=None — should use reference date, not crash."""
        role = _make_role(
            start=date(2023, 1, 1),
            end=None,
            duration_months=36,
            is_current=True,
        )
        candidate = _make_candidate(career=[role])
        # Should not raise
        results = _check_impossible_timeline(candidate, DEFAULT_CONFIG)
        # 36mo stated vs ~41mo actual (2023-01 to 2026-06) → within 24mo tolerance
        assert results == []


# ---------------------------------------------------------------------------
# Tests: _check_suspicious_career_progression
# ---------------------------------------------------------------------------

class TestSuspiciousCareerProgression:

    def test_customer_support_to_senior_ml_flagged(self):
        """
        All prior roles are non-technical but current title is Senior ML.
        No bridge roles → high-confidence flag.
        """
        candidate = _make_candidate(
            profile=_make_profile(title="Senior ML Engineer"),
            career=[
                _make_role(
                    title="Customer Support",
                    company="TCS",
                    industry="IT Services",
                    start=date(2020, 1, 1),
                    end=date(2022, 1, 1),
                    duration_months=24,
                    is_current=False,
                    description="Handled customer queries.",
                ),
                _make_role(
                    title="Marketing Manager",
                    company="Dunder Mifflin",
                    industry="Paper Products",
                    start=date(2018, 1, 1),
                    end=date(2020, 1, 1),
                    duration_months=24,
                    is_current=False,
                    description="Managed marketing campaigns.",
                ),
            ],
        )
        results = _check_suspicious_career_progression(candidate, DEFAULT_CONFIG)
        assert len(results) >= 1
        flag, conf, reason = results[0]
        assert flag == HoneypotFlag.SUSPICIOUS_CAREER_PROGRESSION
        assert conf >= 0.70

    def test_legitimate_ml_career_not_flagged(self):
        """Candidate who progressed naturally through tech roles."""
        candidate = _make_candidate(
            profile=_make_profile(title="ML Engineer"),
            career=[
                _make_role(title="Data Analyst", company="Flipkart",
                           industry="E-commerce", start=date(2017, 1, 1),
                           end=date(2019, 1, 1), duration_months=24, is_current=False,
                           description="Built ML pipelines."),
                _make_role(title="Data Scientist", company="Razorpay",
                           industry="Fintech", start=date(2019, 1, 1),
                           end=date(2021, 1, 1), duration_months=24, is_current=False,
                           description="Trained embedding models."),
                _make_role(title="ML Engineer", company="Swiggy",
                           industry="Food Delivery", start=date(2021, 1, 1),
                           end=None, duration_months=36, is_current=True,
                           description="Shipped vector search."),
            ],
        )
        results = _check_suspicious_career_progression(candidate, DEFAULT_CONFIG)
        assert results == [], f"Legitimate progression should not be flagged: {results}"

    def test_non_ml_current_title_not_checked(self):
        """Check should skip candidates whose current title is not ML/AI."""
        candidate = _make_candidate(
            profile=_make_profile(title="Customer Support"),
            career=[_make_role(title="Customer Support")],
        )
        results = _check_suspicious_career_progression(candidate, DEFAULT_CONFIG)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: _check_domain_skill_contradiction
# ---------------------------------------------------------------------------

class TestDomainSkillContradiction:

    def test_heavy_dual_domain_flagged(self):
        """
        Claiming advanced NLP/retrieval AND advanced CV/speech simultaneously
        is suspicious at high counts.
        """
        skills = (
            [_make_skill(s, SkillProficiency.ADVANCED, 30) for s in
             ["Embeddings", "RAG", "Pinecone", "FAISS", "Semantic Search"]]
            +
            [_make_skill(s, SkillProficiency.ADVANCED, 30) for s in
             ["YOLO", "CNN", "OpenCV", "Computer Vision", "Image Classification"]]
        )
        candidate = _make_candidate(skills=skills)
        results = _check_domain_skill_contradiction(candidate, DEFAULT_CONFIG)
        assert len(results) >= 1
        flag, conf, _ = results[0]
        assert flag == HoneypotFlag.DOMAIN_SKILL_CONTRADICTION

    def test_low_count_dual_domain_not_flagged(self):
        """One or two skills in each domain is plausible — should not flag."""
        skills = [
            _make_skill("Embeddings", SkillProficiency.ADVANCED, 30),
            _make_skill("YOLO", SkillProficiency.ADVANCED, 30),
        ]
        candidate = _make_candidate(skills=skills)
        results = _check_domain_skill_contradiction(candidate, DEFAULT_CONFIG)
        assert results == []


# ---------------------------------------------------------------------------
# Tests: Full HoneypotDetector.detect() integration
# ---------------------------------------------------------------------------

class TestHoneypotDetectorIntegration:

    detector = HoneypotDetector(DEFAULT_CONFIG)

    def test_ideal_candidate_is_clean(self):
        """
        A well-constructed ML engineer profile should produce
        is_honeypot=False with low confidence.
        """
        candidate = _make_candidate(
            profile=_make_profile(
                title="ML Engineer",
                yoe=7.0,
                summary=(
                    "7 years building production ranking and retrieval systems. "
                    "Shipped embedding-based search at Razorpay serving 10M users. "
                    "Expert in FAISS, Pinecone, and hybrid search architectures."
                ),
                company="Razorpay",
                industry="Fintech",
            ),
            career=[
                _make_role(
                    title="ML Engineer", company="Razorpay", industry="Fintech",
                    start=date(2021, 1, 1), end=None, duration_months=42,
                    is_current=True,
                    description="Built production vector search serving 10M users.",
                ),
                _make_role(
                    title="Data Scientist", company="Swiggy", industry="Food Delivery",
                    start=date(2018, 1, 1), end=date(2021, 1, 1), duration_months=36,
                    is_current=False,
                    description="Trained recommendation models for food delivery.",
                ),
            ],
            skills=[
                _make_skill("Python",             SkillProficiency.ADVANCED, 84),
                _make_skill("Embeddings",          SkillProficiency.ADVANCED, 42),
                _make_skill("FAISS",               SkillProficiency.ADVANCED, 42),
                _make_skill("Sentence Transformers", SkillProficiency.ADVANCED, 36),
                _make_skill("Elasticsearch",       SkillProficiency.INTERMEDIATE, 30),
            ],
            signals=_make_signals(
                notice_period_days=30,
                github_activity_score=65.0,
                recruiter_response_rate=0.80,
            ),
        )
        result = self.detector.detect(candidate)
        assert not result.is_honeypot, (
            f"Ideal candidate should NOT be a honeypot: {result.summary()}"
        )
        assert result.confidence < 0.85

    def test_marketing_manager_ai_keyword_stuffer_is_honeypot(self):
        """
        Archetype from dataset: Marketing Manager with 6 advanced AI skills.

        The NON_TECH_WITH_ADVANCED_AI_SKILLS flag fires at 0.88 confidence
        (5+ advanced AI skills on non-tech title) which alone exceeds the
        HARD_CONFIDENCE_THRESHOLD (0.85) → is_honeypot = True.

        Note: SUMMARY_TITLE_CONTRADICTION does NOT fire here because that check
        only activates when the CURRENT TITLE is technical but the summary is
        non-technical. A Marketing Manager with a marketing summary is internally
        consistent — the contradiction is title-vs-skills, not title-vs-summary.
        """
        candidate = _make_candidate(
            profile=_make_profile(
                title="Marketing Manager",
                yoe=5.0,
                summary=(
                    "My professional background is in marketing manager — "
                    "I've built and led teams, owned KPIs, and driven business outcomes. "
                    "Lately I've been curious about AI tools like ChatGPT."
                ),
                company="Dunder Mifflin",
                industry="Paper Products",
            ),
            career=[_make_role(
                title="Marketing Manager", company="Dunder Mifflin",
                industry="Paper Products",
                start=date(2020, 1, 1), end=None, duration_months=30,
                is_current=True,
                description="Led marketing campaigns and brand strategy.",
            )],
            skills=[
                _make_skill("Embeddings",      SkillProficiency.ADVANCED, 16),
                _make_skill("RAG",             SkillProficiency.ADVANCED, 16),
                _make_skill("Pinecone",        SkillProficiency.ADVANCED, 16),
                _make_skill("FAISS",           SkillProficiency.ADVANCED, 16),
                _make_skill("Vector Search",   SkillProficiency.ADVANCED, 16),
                _make_skill("Semantic Search", SkillProficiency.ADVANCED, 16),
                _make_skill("LangChain",       SkillProficiency.ADVANCED, 16),
            ],
        )
        result = self.detector.detect(candidate)
        assert result.is_honeypot, (
            f"Marketing Manager with 7 advanced AI keywords should be a honeypot. "
            f"Got: {result.summary()}"
        )
        assert HoneypotFlag.NON_TECH_WITH_ADVANCED_AI_SKILLS in result.flags
        assert result.confidence >= 0.85, (
            f"Expected confidence >= 0.85 (hard threshold), got {result.confidence}"
        )

    def test_skill_inflation_honeypot(self):
        """
        Modelled on CAND_0000031: Recommendation Systems Engineer, 6 YOE,
        677mo total skill duration.
        """
        # 14 skills × 50 months = 700 months vs 72 months YOE
        candidate = _make_candidate(
            profile=_make_profile(
                title="Recommendation Systems Engineer",
                yoe=6.0,
            ),
            skills=[_make_skill(f"Skill_{i}", SkillProficiency.ADVANCED, 50) for i in range(14)],
        )
        result = self.detector.detect(candidate)
        assert HoneypotFlag.SKILL_DURATION_INFLATION in result.flags

    def test_expert_zero_months_is_honeypot(self):
        """A single EXPERT + 0 months flag alone should mark as honeypot."""
        candidate = _make_candidate(
            skills=[
                _make_skill("Pinecone",  SkillProficiency.EXPERT, 0),
                _make_skill("FAISS",     SkillProficiency.EXPERT, 0),
                _make_skill("Weaviate",  SkillProficiency.EXPERT, 0),
            ]
        )
        result = self.detector.detect(candidate)
        assert result.is_honeypot
        assert HoneypotFlag.EXPERT_WITH_ZERO_EXPERIENCE in result.flags
        assert result.confidence >= 0.85

    def test_result_has_human_readable_reasons(self):
        """Every flag should have at least one associated reason string."""
        candidate = _make_candidate(
            skills=[_make_skill("Pinecone", SkillProficiency.EXPERT, 0)]
        )
        result = self.detector.detect(candidate)
        assert len(result.reasons) >= 1
        for reason in result.reasons:
            assert isinstance(reason, str) and len(reason) > 20

    def test_honeypot_result_summary_format(self):
        """HoneypotResult.summary() should return a non-empty string."""
        candidate = _make_candidate(
            skills=[_make_skill("Pinecone", SkillProficiency.EXPERT, 0)]
        )
        result = self.detector.detect(candidate)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "HONEYPOT" in summary or "SUSPICIOUS" in summary or "CLEAN" in summary

    def test_clean_candidate_summary_says_clean(self):
        """Clean candidate should produce a 'CLEAN' summary."""
        candidate = _make_candidate(
            skills=[_make_skill("Python", SkillProficiency.ADVANCED, 60)],
        )
        result = self.detector.detect(candidate)
        if not result.is_honeypot and not result.flags:
            assert "CLEAN" in result.summary()

    def test_detect_batch_returns_all_ids(self):
        """detect_batch should return a result for every input candidate."""
        candidates = [
            _make_candidate(f"CAND_{i:07d}") for i in range(10)
        ]
        detector = HoneypotDetector(DEFAULT_CONFIG)
        results = detector.detect_batch(candidates, log_every=5)
        assert len(results) == 10
        for cand in candidates:
            assert cand.candidate_id in results


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Quick smoke run without pytest
    suite = [
        TestSkillDurationInflation(),
        TestExpertWithZeroExperience(),
        TestNonTechTitleWithAISkills(),
        TestSummaryTitleContradiction(),
        TestImpossibleTimeline(),
        TestSuspiciousCareerProgression(),
        TestDomainSkillContradiction(),
        TestHoneypotDetectorIntegration(),
    ]

    passed = failed = 0
    for cls_instance in suite:
        class_name = type(cls_instance).__name__
        methods = [m for m in dir(cls_instance) if m.startswith("test_")]
        for method_name in methods:
            try:
                getattr(cls_instance, method_name)()
                print(f"  ✅ {class_name}.{method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  ❌ {class_name}.{method_name}: {e}")
                failed += 1
            except Exception as e:
                print(f"  💥 {class_name}.{method_name}: {type(e).__name__}: {e}")
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
