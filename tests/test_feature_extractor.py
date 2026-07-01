"""
tests/test_feature_extractor.py
================================
Unit tests for FeatureExtractor and its helper functions.

Each test class targets one extraction group or helper.
Integration tests at the bottom use full realistic candidate profiles
modelled on the dataset archetypes identified in dataset_profile.md.

Archetypes tested
-----------------
1. Ideal ML Engineer (product company, retrieval background)
2. Marketing Manager keyword stuffer (should produce low relevance scores)
3. Pure IT-services engineer (Infosys/TCS background only)
4. CV/Speech specialist (wrong domain)
5. Data scientist with mixed background
6. Honeypot candidate (inflated signals from detector)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
from typing import Optional

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
from src.scoring.feature_extractor import (
    FeatureExtractor,
    FeatureVector,
    TitleCategory,
    classify_title,
    _technical_title_score,
    _leadership_score,
)
from src.scoring.honeypot_detector import HoneypotDetector, HoneypotResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _signals(**overrides) -> RedrobSignals:
    defaults = dict(
        profile_completeness_score=80.0,
        signup_date=date(2025, 1, 1),
        last_active_date=date(2026, 5, 15),
        open_to_work_flag=True,
        profile_views_received_30d=30,
        applications_submitted_30d=3,
        recruiter_response_rate=0.75,
        avg_response_time_hours=12.0,
        skill_assessment_scores={},
        connection_count=400,
        endorsements_received=35,
        notice_period_days=30,
        expected_salary_range_inr_lpa=SalaryRange(30.0, 50.0),
        preferred_work_mode=WorkMode.HYBRID,
        willing_to_relocate=True,
        github_activity_score=60.0,
        search_appearance_30d=120,
        saved_by_recruiters_30d=8,
        interview_completion_rate=0.90,
        offer_acceptance_rate=0.75,
        verified_email=True,
        verified_phone=True,
        linkedin_connected=True,
    )
    defaults.update(overrides)
    return RedrobSignals(**defaults)


def _role(
    title: str = "ML Engineer",
    company: str = "Razorpay",
    industry: str = "Fintech",
    start: date = date(2021, 1, 1),
    end: Optional[date] = None,
    duration_months: int = 36,
    is_current: bool = True,
    description: str = "Built vector search and ranking systems.",
) -> CareerRole:
    return CareerRole(
        company=company, title=title, start_date=start, end_date=end,
        duration_months=duration_months, is_current=is_current,
        industry=industry, company_size=CompanySize.XL, description=description,
    )


def _skill(
    name: str,
    prof: SkillProficiency = SkillProficiency.ADVANCED,
    duration: int = 24,
    endorsements: int = 10,
) -> Skill:
    return Skill(name=name, proficiency=prof, endorsements=endorsements, duration_months=duration)


def _profile(
    title: str = "ML Engineer",
    yoe: float = 7.0,
    company: str = "Razorpay",
    industry: str = "Fintech",
    summary: str = "ML engineer building retrieval systems.",
) -> CandidateProfile:
    return CandidateProfile(
        anonymized_name="Test", headline=title, summary=summary,
        location="Bangalore", country="India",
        years_of_experience=yoe, current_title=title,
        current_company=company, current_company_size=CompanySize.XL,
        current_industry=industry,
    )


def _make_candidate(
    candidate_id: str = "CAND_TEST",
    profile: Optional[CandidateProfile] = None,
    career: Optional[list[CareerRole]] = None,
    skills: Optional[list[Skill]] = None,
    signals: Optional[RedrobSignals] = None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        profile=profile or _profile(),
        career_history=career or [_role()],
        education=[Education("IIT", "B.Tech", "CS", 2012, 2016, "9.0", InstitutionTier.TIER_1)],
        skills=skills or [_skill("Python"), _skill("Embeddings")],
        certifications=[], languages=[Language("English", LanguageProficiency.PROFESSIONAL)],
        signals=signals or _signals(),
    )


EXT = FeatureExtractor(DEFAULT_CONFIG)
DET = HoneypotDetector(DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Tests: classify_title()
# ---------------------------------------------------------------------------

class TestClassifyTitle:

    def test_ai_engineer_variants(self):
        cases = [
            "AI Engineer", "Senior AI Engineer", "Lead AI Engineer",
            "NLP Engineer", "Search Engineer", "Applied ML Engineer",
            "Recommendation Systems Engineer", "Retrieval Engineer",
        ]
        for title in cases:
            cat = classify_title(title)
            assert cat == TitleCategory.AI_ENGINEER, f"Expected AI_ENGINEER for '{title}', got {cat}"

    def test_ml_engineer_variants(self):
        cases = [
            "ML Engineer", "Machine Learning Engineer", "Senior ML Engineer",
            "Staff Machine Learning Engineer", "Junior ML Engineer",
            "Computer Vision Engineer",
        ]
        for title in cases:
            cat = classify_title(title)
            assert cat == TitleCategory.ML_ENGINEER, f"Expected ML_ENGINEER for '{title}', got {cat}"

    def test_data_scientist_variants(self):
        cases = ["Data Scientist", "Senior Data Scientist", "AI Research Engineer"]
        for title in cases:
            cat = classify_title(title)
            assert cat in (TitleCategory.DATA_SCIENTIST, TitleCategory.AI_ENGINEER), \
                f"Unexpected category for '{title}': {cat}"

    def test_data_engineer_variants(self):
        cases = ["Data Engineer", "Analytics Engineer", "Senior Data Engineer"]
        for title in cases:
            cat = classify_title(title)
            assert cat == TitleCategory.DATA_ENGINEER, f"Expected DATA_ENGINEER for '{title}', got {cat}"

    def test_software_engineer_variants(self):
        cases = [
            "Software Engineer", "Backend Engineer", "Frontend Engineer",
            "Full Stack Developer", "Java Developer", ".NET Developer",
            "Mobile Developer", "QA Engineer",
        ]
        for title in cases:
            cat = classify_title(title)
            assert cat == TitleCategory.SOFTWARE_ENGINEER, \
                f"Expected SOFTWARE_ENGINEER for '{title}', got {cat}"

    def test_non_technical_titles(self):
        cases = [
            "Marketing Manager", "HR Manager", "Accountant",
            "Customer Support", "Sales Executive", "Content Writer",
            "Graphic Designer", "Operations Manager",
        ]
        for title in cases:
            cat = classify_title(title)
            assert cat == TitleCategory.NON_TECHNICAL, \
                f"Expected NON_TECHNICAL for '{title}', got {cat}"

    def test_manager_category(self):
        cases = ["Project Manager", "Engineering Manager", "VP of Engineering"]
        for title in cases:
            cat = classify_title(title)
            assert cat in (TitleCategory.MANAGER, TitleCategory.NON_TECHNICAL), \
                f"Unexpected for '{title}': {cat}"


# ---------------------------------------------------------------------------
# Tests: technical_title_score and leadership_score
# ---------------------------------------------------------------------------

class TestTitleScores:

    def test_ai_engineer_has_max_score(self):
        assert _technical_title_score("NLP Engineer") == 1.00

    def test_non_technical_has_zero_score(self):
        assert _technical_title_score("Marketing Manager") == 0.00
        assert _technical_title_score("HR Manager") == 0.00
        assert _technical_title_score("Accountant") == 0.00

    def test_software_engineer_mid_score(self):
        score = _technical_title_score("Software Engineer")
        assert 0.40 <= score <= 0.60

    def test_staff_leadership_highest(self):
        assert _leadership_score("Staff ML Engineer") >= 0.85

    def test_senior_leadership_mid_high(self):
        score = _leadership_score("Senior ML Engineer")
        assert 0.60 <= score <= 0.80

    def test_junior_leadership_low(self):
        assert _leadership_score("Junior ML Engineer") <= 0.30


# ---------------------------------------------------------------------------
# Tests: _extract_experience
# ---------------------------------------------------------------------------

class TestExtractExperience:

    def test_yoe_copied_correctly(self):
        candidate = _make_candidate(profile=_profile(yoe=7.5))
        fv = EXT.extract(candidate)
        assert fv.years_experience == 7.5

    def test_total_roles_counted(self):
        candidate = _make_candidate(career=[
            _role(title="Data Scientist", company="Swiggy", is_current=False,
                  end=date(2023, 1, 1), duration_months=24),
            _role(title="ML Engineer", company="Razorpay", duration_months=36),
        ])
        fv = EXT.extract(candidate)
        assert fv.total_roles == 2

    def test_avg_tenure_calculated(self):
        candidate = _make_candidate(career=[
            _role(duration_months=24, is_current=False, end=date(2022,1,1)),
            _role(duration_months=36),
        ])
        fv = EXT.extract(candidate)
        assert fv.avg_tenure_months == 30.0

    def test_single_role_avg_tenure(self):
        candidate = _make_candidate(career=[_role(duration_months=48)])
        fv = EXT.extract(candidate)
        assert fv.avg_tenure_months == 48.0


# ---------------------------------------------------------------------------
# Tests: _extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:

    def test_title_category_set(self):
        candidate = _make_candidate(profile=_profile(title="ML Engineer"))
        fv = EXT.extract(candidate)
        assert fv.title_category == TitleCategory.ML_ENGINEER.value

    def test_non_tech_title_zero_score(self):
        candidate = _make_candidate(profile=_profile(title="Marketing Manager"))
        fv = EXT.extract(candidate)
        assert fv.technical_title_score == 0.0
        assert fv.title_category in (
            TitleCategory.NON_TECHNICAL.value, TitleCategory.MANAGER.value
        ), f"Got unexpected category: {fv.title_category}"

    def test_current_title_stored(self):
        candidate = _make_candidate(profile=_profile(title="Search Engineer"))
        fv = EXT.extract(candidate)
        assert fv.current_title == "Search Engineer"


# ---------------------------------------------------------------------------
# Tests: _extract_companies
# ---------------------------------------------------------------------------

class TestExtractCompanies:

    def test_product_company_detected(self):
        candidate = _make_candidate(career=[
            _role(company="Razorpay", industry="Fintech"),
            _role(company="Swiggy",   industry="Food Delivery", is_current=False,
                  end=date(2021,1,1), duration_months=24),
        ])
        fv = EXT.extract(candidate)
        assert fv.product_company_count == 2
        assert fv.service_company_count == 0
        assert fv.product_company_ratio == 1.0

    def test_services_company_detected(self):
        candidate = _make_candidate(career=[
            _role(company="Infosys", industry="IT Services"),
            _role(company="Wipro",   industry="IT Services", is_current=False,
                  end=date(2021,1,1), duration_months=24),
        ])
        fv = EXT.extract(candidate)
        assert fv.service_company_count == 2
        assert fv.product_company_count == 0
        assert fv.product_company_ratio == 0.0

    def test_mixed_companies(self):
        candidate = _make_candidate(career=[
            _role(company="TCS",     industry="IT Services",  is_current=False,
                  end=date(2020,1,1), duration_months=24),
            _role(company="Swiggy",  industry="Food Delivery"),
        ])
        fv = EXT.extract(candidate)
        assert fv.product_company_count == 1
        assert fv.service_company_count == 1
        assert fv.product_company_ratio == 0.5


# ---------------------------------------------------------------------------
# Tests: _extract_skills
# ---------------------------------------------------------------------------

class TestExtractSkills:

    def test_retrieval_skills_counted(self):
        candidate = _make_candidate(skills=[
            _skill("FAISS"),
            _skill("Elasticsearch"),
            _skill("Semantic Search"),
        ])
        fv = EXT.extract(candidate)
        assert fv.retrieval_skill_count >= 3

    def test_vector_db_skills_counted(self):
        candidate = _make_candidate(skills=[
            _skill("Pinecone"),
            _skill("Milvus"),
            _skill("Qdrant"),
        ])
        fv = EXT.extract(candidate)
        assert fv.vector_db_skill_count >= 3

    def test_llm_skills_counted(self):
        candidate = _make_candidate(skills=[
            _skill("RAG"),
            _skill("Embeddings"),
            _skill("LangChain"),
        ])
        fv = EXT.extract(candidate)
        assert fv.llm_skill_count >= 3

    def test_cv_skills_counted(self):
        candidate = _make_candidate(skills=[
            _skill("YOLO"),
            _skill("CNN"),
            _skill("OpenCV"),
        ])
        fv = EXT.extract(candidate)
        assert fv.cv_skill_count >= 3
        # CV skills should NOT count as retrieval
        assert fv.retrieval_skill_count == 0

    def test_advanced_expert_counts(self):
        candidate = _make_candidate(skills=[
            _skill("PyTorch",   SkillProficiency.EXPERT),
            _skill("Python",    SkillProficiency.ADVANCED),
            _skill("Spark",     SkillProficiency.INTERMEDIATE),
        ])
        fv = EXT.extract(candidate)
        assert fv.expert_skill_count == 1
        assert fv.advanced_skill_count == 1

    def test_avg_skill_duration(self):
        candidate = _make_candidate(skills=[
            _skill("Python", duration=60),
            _skill("FAISS",  duration=30),
        ])
        fv = EXT.extract(candidate)
        assert fv.avg_skill_duration == 45.0

    def test_empty_skills(self):
        """With no skills, all skill counts and avg duration should be zero."""
        # Use CandidateRecord directly to guarantee empty skills list
        from datetime import date as d
        candidate = CandidateRecord(
            candidate_id="CAND_EMPTY_SKILLS",
            profile=_profile(title="ML Engineer"),
            career_history=[_role()],
            education=[], certifications=[], languages=[],
            skills=[],   # explicitly empty
            signals=_signals(),
        )
        fv = EXT.extract(candidate)
        assert fv.retrieval_skill_count == 0
        assert fv.avg_skill_duration    == 0.0
        assert fv.advanced_skill_count  == 0

    def test_faiss_counts_in_both_retrieval_and_vector_db(self):
        """FAISS is both a retrieval tool and a vector DB — should count in both."""
        candidate = _make_candidate(skills=[_skill("FAISS")])
        fv = EXT.extract(candidate)
        assert fv.retrieval_skill_count >= 1
        assert fv.vector_db_skill_count >= 1


# ---------------------------------------------------------------------------
# Tests: _extract_behavioral
# ---------------------------------------------------------------------------

class TestExtractBehavioral:

    def test_all_behavioral_signals_copied(self):
        sigs = _signals(
            recruiter_response_rate=0.80,
            avg_response_time_hours=8.0,
            github_activity_score=70.0,
            open_to_work_flag=True,
            notice_period_days=15,
            interview_completion_rate=0.95,
            offer_acceptance_rate=0.60,
        )
        candidate = _make_candidate(signals=sigs)
        fv = EXT.extract(candidate)

        assert fv.recruiter_response_rate   == 0.80
        assert fv.avg_response_time_hours   == 8.0
        assert fv.github_activity_score     == 70.0
        assert fv.open_to_work              is True
        assert fv.notice_period_days        == 15
        assert fv.interview_completion_rate == 0.95
        assert fv.offer_acceptance_rate     == 0.60

    def test_no_github_sentinel_preserved(self):
        sigs = _signals(github_activity_score=-1.0)
        candidate = _make_candidate(signals=sigs)
        fv = EXT.extract(candidate)
        assert fv.github_activity_score == -1.0


# ---------------------------------------------------------------------------
# Tests: _extract_assessments
# ---------------------------------------------------------------------------

class TestExtractAssessments:

    def test_skill_assessments_averaged(self):
        sigs = _signals(skill_assessment_scores={
            "Python": 85.0,
            "Elasticsearch": 75.0,
            "FAISS": 90.0,
        })
        candidate = _make_candidate(signals=sigs)
        fv = EXT.extract(candidate)
        assert fv.has_skill_assessments is True
        assert abs(fv.average_skill_assessment - 83.33) < 0.1

    def test_empty_assessments(self):
        sigs = _signals(skill_assessment_scores={})
        candidate = _make_candidate(signals=sigs)
        fv = EXT.extract(candidate)
        assert fv.has_skill_assessments    is False
        assert fv.average_skill_assessment == 0.0


# ---------------------------------------------------------------------------
# Tests: _extract_career_evidence
# ---------------------------------------------------------------------------

class TestExtractCareerEvidence:

    def test_retrieval_evidence_from_description(self):
        candidate = _make_candidate(career=[
            _role(description=(
                "Built a semantic search engine using FAISS and Elasticsearch. "
                "Implemented hybrid search combining BM25 and dense retrieval. "
                "Designed the retrieval pipeline serving 5M daily queries."
            )),
        ])
        fv = EXT.extract(candidate)
        assert fv.retrieval_evidence_count >= 5, \
            f"Expected >=5 retrieval mentions, got {fv.retrieval_evidence_count}"

    def test_ranking_evidence_from_description(self):
        candidate = _make_candidate(career=[
            _role(description=(
                "Implemented a ranking model using learning to rank with NDCG "
                "offline evaluation. The ranking layer re-ranked top-100 candidates "
                "from the retrieval stage."
            )),
        ])
        fv = EXT.extract(candidate)
        assert fv.ranking_evidence_count >= 3

    def test_vector_db_evidence(self):
        candidate = _make_candidate(career=[
            _role(description=(
                "Deployed Pinecone for production vector index serving 10M embeddings. "
                "Migrated from FAISS to Qdrant for better recall. "
                "Evaluated Milvus and Weaviate as vector database alternatives."
            )),
        ])
        fv = EXT.extract(candidate)
        assert fv.vector_db_evidence_count >= 4

    def test_no_evidence_in_non_tech_description(self):
        candidate = _make_candidate(career=[
            _role(description=(
                "Led marketing campaigns for the brand. Managed a team of 5 "
                "content writers. Owned the editorial calendar and KPI dashboard."
            )),
        ])
        fv = EXT.extract(candidate)
        assert fv.retrieval_evidence_count  == 0
        assert fv.vector_db_evidence_count  == 0
        assert fv.ranking_evidence_count    == 0

    def test_production_evidence_counted(self):
        candidate = _make_candidate(career=[
            _role(description=(
                "Shipped a production ranking system serving millions of users. "
                "Set up A/B testing infrastructure for online evaluation. "
                "Optimised inference latency from 200ms to 40ms."
            )),
        ])
        fv = EXT.extract(candidate)
        assert fv.ml_production_evidence_count >= 3

    def test_career_text_length_positive(self):
        candidate = _make_candidate(career=[
            _role(description="Built production ML pipelines and ranking systems.")
        ])
        fv = EXT.extract(candidate)
        assert fv.career_text_length > 0


# ---------------------------------------------------------------------------
# Tests: _extract_career_quality
# ---------------------------------------------------------------------------

class TestExtractCareerQuality:

    def test_product_ml_years_detected(self):
        """
        ML Engineer at Razorpay (product company + ML title) should accumulate
        product_ml_experience_years.
        """
        candidate = _make_candidate(
            career=[
                _role(title="ML Engineer", company="Razorpay", industry="Fintech",
                      duration_months=48),
            ]
        )
        fv = EXT.extract(candidate)
        assert fv.product_ml_experience_years == 4.0, \
            f"Expected 4.0 years, got {fv.product_ml_experience_years}"

    def test_services_ml_not_counted_in_product_ml(self):
        """ML Engineer at Infosys (services) should NOT count as product ML."""
        candidate = _make_candidate(
            career=[
                _role(title="ML Engineer", company="Infosys", industry="IT Services",
                      duration_months=36),
            ]
        )
        fv = EXT.extract(candidate)
        assert fv.product_ml_experience_years == 0.0, \
            f"Services ML should not count as product ML, got {fv.product_ml_experience_years}"

    def test_ml_description_at_product_company_counts(self):
        """
        Even without 'ML' in the title, if the description has strong ML evidence
        at a product company, it should count.
        """
        candidate = _make_candidate(
            career=[
                _role(
                    title="Software Engineer",
                    company="Swiggy",
                    industry="Food Delivery",
                    duration_months=24,
                    description=(
                        "Built recommendation model using embeddings and vector search. "
                        "Trained neural ranking model and deployed to inference. "
                        "Used sklearn and PyTorch for ML pipeline."
                    ),
                ),
            ]
        )
        fv = EXT.extract(candidate)
        assert fv.product_ml_experience_years > 0, \
            "ML description at product company should accumulate product_ml_years"

    def test_technical_role_ratio_all_technical(self):
        candidate = _make_candidate(career=[
            _role(title="Data Scientist", company="Swiggy", is_current=False,
                  end=date(2022,1,1), duration_months=24),
            _role(title="ML Engineer", company="Razorpay"),
        ])
        fv = EXT.extract(candidate)
        assert fv.technical_role_ratio == 1.0

    def test_technical_role_ratio_mixed(self):
        candidate = _make_candidate(career=[
            _role(title="Marketing Manager", company="Dunder Mifflin",
                  industry="Paper Products", is_current=False,
                  end=date(2020,1,1), duration_months=24,
                  description="Ran marketing campaigns."),
            _role(title="ML Engineer", company="Razorpay"),
        ])
        fv = EXT.extract(candidate)
        assert 0.0 < fv.technical_role_ratio < 1.0

    def test_technical_role_ratio_no_career(self):
        """With empty career history, ratio should be 0.0, not 1.0."""
        # Use CandidateRecord directly to guarantee empty career list
        candidate = CandidateRecord(
            candidate_id="CAND_EMPTY_CAREER",
            profile=_profile(title="ML Engineer"),
            career_history=[],   # explicitly empty
            education=[], certifications=[], languages=[],
            skills=[_skill("Python")],
            signals=_signals(),
        )
        fv = EXT.extract(candidate)
        assert fv.technical_role_ratio == 0.0
        assert fv.total_roles          == 0


# ---------------------------------------------------------------------------
# Tests: _extract_honeypot
# ---------------------------------------------------------------------------

class TestExtractHoneypot:

    def test_none_honeypot_zeroed(self):
        candidate = _make_candidate()
        fv = EXT.extract(candidate, honeypot_result=None)
        assert fv.honeypot_confidence == 0.0
        assert fv.honeypot_flag_count  == 0

    def test_honeypot_confidence_passed_through(self):
        from src.scoring.honeypot_detector import HoneypotFlag
        hp = HoneypotResult(
            is_honeypot=True, confidence=0.92,
            flags=[HoneypotFlag.EXPERT_WITH_ZERO_EXPERIENCE,
                   HoneypotFlag.SKILL_DURATION_INFLATION],
            reasons=["reason1", "reason2"],
            flag_confidences={},
        )
        candidate = _make_candidate()
        fv = EXT.extract(candidate, honeypot_result=hp)
        assert fv.honeypot_confidence == 0.92
        assert fv.honeypot_flag_count  == 2


# ---------------------------------------------------------------------------
# Integration tests — full archetype profiles
# ---------------------------------------------------------------------------

class TestFullArchetypes:

    def _hp(self, candidate: CandidateRecord) -> HoneypotResult:
        return DET.detect(candidate)

    def test_ideal_ml_engineer_features(self):
        """
        Archetype: 7 YOE ML Engineer at product companies with retrieval background.
        Expected: high scores across all relevant feature groups.
        """
        candidate = _make_candidate(
            candidate_id="CAND_IDEAL",
            profile=_profile(
                title="ML Engineer", yoe=7.0,
                company="Razorpay", industry="Fintech",
                summary="7 years building production ranking and retrieval systems.",
            ),
            career=[
                _role(
                    title="ML Engineer", company="Razorpay", industry="Fintech",
                    duration_months=42, is_current=True,
                    description=(
                        "Shipped production semantic search using FAISS and Pinecone. "
                        "Designed hybrid retrieval combining BM25 and dense retrieval techniques. "
                        "Implemented learning to rank with NDCG-optimised training. "
                        "Deployed recommendation system serving 10M users. "
                        "Optimized search relevance across the information retrieval pipeline. "
                        "Set up A/B testing infrastructure for online ranking evaluation."
                    ),
                ),
                _role(
                    title="Data Scientist", company="Swiggy", industry="Food Delivery",
                    start=date(2018, 1, 1), end=date(2021, 1, 1),
                    duration_months=36, is_current=False,
                    description=(
                        "Trained recommendation models using PyTorch. "
                        "Developed collaborative filtering models to improve engagement. "
                        "Built vector search pipeline for product discovery. "
                        "Deployed inference service with 40ms p99 latency."
                    ),
                ),
            ],
            skills=[
                _skill("Python",                 SkillProficiency.ADVANCED, 84),
                _skill("Embeddings",             SkillProficiency.ADVANCED, 42),
                _skill("FAISS",                  SkillProficiency.ADVANCED, 42),
                _skill("Sentence Transformers",  SkillProficiency.ADVANCED, 36),
                _skill("Elasticsearch",          SkillProficiency.ADVANCED, 36),
                _skill("Pinecone",               SkillProficiency.ADVANCED, 24),
                _skill("Recommendation Systems", SkillProficiency.ADVANCED, 30),
                _skill("Learning to Rank",       SkillProficiency.INTERMEDIATE, 24),
                _skill("PyTorch",                SkillProficiency.ADVANCED, 60),
            ],
            signals=_signals(
                open_to_work_flag=True,
                notice_period_days=30,
                recruiter_response_rate=0.85,
                github_activity_score=72.0,
                interview_completion_rate=0.95,
            ),
        )
        fv = EXT.extract(candidate, self._hp(candidate))

        # Title
        assert fv.title_category == TitleCategory.ML_ENGINEER.value
        assert fv.technical_title_score >= 0.90

        # Company quality
        assert fv.product_company_count >= 2
        assert fv.product_company_ratio == 1.0

        # Skill signals
        assert fv.retrieval_skill_count  >= 3
        assert fv.vector_db_skill_count  >= 2
        assert fv.llm_skill_count        >= 2

        # Career evidence
        assert fv.retrieval_evidence_count      >= 5
        assert fv.ranking_evidence_count        >= 2
        assert fv.recommendation_evidence_count >= 2
        assert fv.vector_db_evidence_count      >= 2

        # Product ML YOE
        assert fv.product_ml_experience_years >= 5.0

        # Behavioral
        assert fv.open_to_work              is True
        assert fv.notice_period_days        == 30
        assert fv.recruiter_response_rate   >= 0.80

        # Not a honeypot
        assert fv.honeypot_confidence < 0.50

    def test_marketing_manager_keyword_stuffer(self):
        """
        Archetype: Marketing Manager with AI keywords stuffed into skills.
        Expected: high honeypot confidence, zero product ML years, zero retrieval evidence.
        """
        candidate = _make_candidate(
            candidate_id="CAND_STUFFER",
            profile=_profile(
                title="Marketing Manager", yoe=5.0,
                company="Dunder Mifflin", industry="Paper Products",
            ),
            career=[
                _role(
                    title="Marketing Manager", company="Dunder Mifflin",
                    industry="Paper Products", duration_months=60,
                    description="Ran marketing campaigns and managed brand strategy.",
                ),
            ],
            skills=[
                _skill("Embeddings",      SkillProficiency.ADVANCED, 16),
                _skill("RAG",             SkillProficiency.ADVANCED, 16),
                _skill("Pinecone",        SkillProficiency.ADVANCED, 16),
                _skill("Vector Search",   SkillProficiency.ADVANCED, 16),
                _skill("Semantic Search", SkillProficiency.ADVANCED, 16),
                _skill("LangChain",       SkillProficiency.ADVANCED, 16),
            ],
        )
        fv = EXT.extract(candidate, self._hp(candidate))

        assert fv.title_category       == TitleCategory.NON_TECHNICAL.value
        assert fv.technical_title_score == 0.0
        assert fv.product_company_count == 0
        assert fv.product_ml_experience_years == 0.0
        assert fv.retrieval_evidence_count     == 0
        assert fv.honeypot_confidence          >= 0.85   # hard threshold exceeded

    def test_pure_services_engineer(self):
        """
        Archetype: Entire career at TCS/Infosys — no product company experience.
        Expected: zero product ML years, zero product company ratio.
        """
        candidate = _make_candidate(
            candidate_id="CAND_SERVICES",
            profile=_profile(title="Software Engineer", yoe=8.0, company="TCS"),
            career=[
                _role(title="Software Engineer", company="TCS",
                      industry="IT Services", duration_months=48),
                _role(title="Developer", company="Infosys",
                      industry="IT Services", is_current=False,
                      end=date(2022, 1, 1), duration_months=48),
            ],
            skills=[_skill("Java"), _skill("Spring Boot"), _skill("SQL")],
        )
        fv = EXT.extract(candidate, self._hp(candidate))

        assert fv.product_company_ratio      == 0.0
        assert fv.product_ml_experience_years == 0.0
        assert fv.retrieval_skill_count        == 0

    def test_cv_speech_specialist(self):
        """
        Archetype: Computer Vision specialist — wrong domain for this JD.
        Expected: high cv_skill_count, low retrieval/vector_db counts.
        """
        candidate = _make_candidate(
            candidate_id="CAND_CV",
            profile=_profile(title="Computer Vision Engineer", yoe=6.0),
            career=[
                _role(
                    title="Computer Vision Engineer",
                    company="Hooli", industry="Software",
                    description=(
                        "Built image classification pipeline using CNN and ResNet. "
                        "Deployed object detection with YOLO. "
                        "Used OpenCV for image preprocessing."
                    ),
                ),
            ],
            skills=[
                _skill("YOLO",              SkillProficiency.ADVANCED, 36),
                _skill("CNN",               SkillProficiency.ADVANCED, 36),
                _skill("OpenCV",            SkillProficiency.ADVANCED, 36),
                _skill("Image Classification", SkillProficiency.ADVANCED, 36),
                _skill("Computer Vision",   SkillProficiency.ADVANCED, 36),
            ],
        )
        fv = EXT.extract(candidate, self._hp(candidate))

        assert fv.cv_skill_count          >= 4
        assert fv.retrieval_skill_count    == 0
        assert fv.vector_db_skill_count    == 0
        assert fv.product_ml_experience_years < 1.0   # Hooli is product but no ML evidence

    def test_feature_vector_all_fields_present(self):
        """FeatureVector should never have missing fields."""
        candidate = _make_candidate()
        fv = EXT.extract(candidate)

        # All numeric fields should be numeric types, not None
        for fname, fdef in FeatureVector.__dataclass_fields__.items():
            val = getattr(fv, fname)
            if fname == "candidate_id":
                assert isinstance(val, str)
            elif fname in ("open_to_work", "has_skill_assessments"):
                assert isinstance(val, bool)
            elif fname in ("total_roles", "notice_period_days",
                           "retrieval_skill_count", "vector_db_skill_count",
                           "llm_skill_count", "ranking_skill_count",
                           "recommendation_skill_count", "ml_skill_count",
                           "cv_skill_count", "advanced_skill_count",
                           "expert_skill_count", "product_company_count",
                           "service_company_count", "honeypot_flag_count",
                           "retrieval_evidence_count", "ranking_evidence_count",
                           "recommendation_evidence_count", "vector_db_evidence_count",
                           "ml_production_evidence_count", "career_text_length"):
                assert isinstance(val, int), f"Field {fname} should be int, got {type(val)}"
            elif fname == "top_domain_skills":
                assert isinstance(val, list)
            else:
                assert isinstance(val, (int, float, str, bool)), \
                    f"Field {fname} has unexpected type {type(val)}"

    def test_extract_batch_parallel_to_extract(self):
        """extract_batch should return same results as individual extract() calls."""
        candidates = [
            _make_candidate(f"CAND_{i:04d}") for i in range(5)
        ]
        individual = [EXT.extract(c) for c in candidates]
        batch      = EXT.extract_batch(candidates)

        for ind, bat in zip(individual, batch):
            assert ind.candidate_id              == bat.candidate_id
            assert ind.product_company_ratio     == bat.product_company_ratio
            assert ind.retrieval_evidence_count  == bat.retrieval_evidence_count


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_classes = [
        TestClassifyTitle(),
        TestTitleScores(),
        TestExtractExperience(),
        TestExtractTitle(),
        TestExtractCompanies(),
        TestExtractSkills(),
        TestExtractBehavioral(),
        TestExtractAssessments(),
        TestExtractCareerEvidence(),
        TestExtractCareerQuality(),
        TestExtractHoneypot(),
        TestFullArchetypes(),
    ]

    passed = failed = 0
    for cls_instance in test_classes:
        class_name = type(cls_instance).__name__
        methods = sorted(m for m in dir(cls_instance) if m.startswith("test_"))
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
                import traceback; traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    import sys; sys.exit(0 if failed == 0 else 1)
