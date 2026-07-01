"""
src/scoring/feature_extractor.py
==================================
Converts a CandidateRecord + HoneypotResult into a flat FeatureVector
containing every signal the scoring engine will need.

Design principles
-----------------
- Pure extraction: no scoring, no ranking, no final decisions here.
- All output fields are primitives (float, int, bool, str) so the scoring
  engine operates on simple arithmetic, not nested objects.
- All keyword sets are module-level frozensets — one source of truth,
  no repetition across check methods.
- Career evidence uses str.count() — multiple mentions accumulate signal.
- Every extraction method is private and independently unit-testable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from config.settings import Config, DEFAULT_CONFIG, CareerConfig
from src.models.candidate import (
    CandidateRecord,
    CareerRole,
    SkillProficiency,
)
from src.scoring.honeypot_detector import HoneypotResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Title category enum
# ---------------------------------------------------------------------------

class TitleCategory(str, Enum):
    """
    Coarse-grained title classification used as a categorical feature.

    Stored as string in FeatureVector so it serialises cleanly to CSV/JSON.
    """
    AI_ENGINEER       = "AI_ENGINEER"
    ML_ENGINEER       = "ML_ENGINEER"
    DATA_SCIENTIST    = "DATA_SCIENTIST"
    DATA_ENGINEER     = "DATA_ENGINEER"
    SOFTWARE_ENGINEER = "SOFTWARE_ENGINEER"
    DEVOPS            = "DEVOPS"
    MANAGER           = "MANAGER"
    NON_TECHNICAL     = "NON_TECHNICAL"
    UNKNOWN           = "UNKNOWN"


# ---------------------------------------------------------------------------
# 2.  Skill keyword groups (sourced from dataset_profile.md EDA)
# ---------------------------------------------------------------------------

# Retrieval / search / ranking infrastructure
RETRIEVAL_SKILLS: frozenset[str] = frozenset({
    "information retrieval", "bm25", "search", "elasticsearch", "opensearch",
    "faiss", "learning to rank", "ltr", "semantic search", "vector search",
    "annoy", "hnswlib", "solr", "information retrieval systems",
    "ranking systems", "search backend", "search & discovery",
    "hybrid search", "dense retrieval", "sparse retrieval",
})

# Vector database platforms
VECTOR_DB_SKILLS: frozenset[str] = frozenset({
    "pinecone", "milvus", "qdrant", "weaviate", "pgvector",
    "faiss", "opensearch", "chroma", "vespa", "vald",
    "lance", "turbopuffer",
})

# LLM / NLP / embedding stack
LLM_SKILLS: frozenset[str] = frozenset({
    "llms", "rag", "embeddings", "sentence transformers",
    "langchain", "fine-tuning llms", "hugging face transformers",
    "prompt engineering", "llamaindex", "llama index",
    "qlora", "lora", "peft", "nlp", "transformers",
    "bert", "gpt", "machine learning", "deep learning",
    "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "openai", "anthropic", "gemini",
})

# Learning-to-rank / ranking models
RANKING_SKILLS: frozenset[str] = frozenset({
    "learning to rank", "ltr", "bm25", "ndcg", "mrr", "map",
    "ranking systems", "recommendation systems",
    "information retrieval", "xgboost", "lightgbm",
    "neural ranking", "pointwise", "pairwise", "listwise",
})

# Recommendation systems
RECOMMENDATION_SKILLS: frozenset[str] = frozenset({
    "recommendation systems", "collaborative filtering",
    "matrix factorization", "svd", "als",
    "content-based filtering", "two-tower", "retrieval augmented",
})

# Broader ML / data science stack
ML_SKILLS: frozenset[str] = frozenset({
    "feature engineering", "mlflow", "mlops",
    "weights & biases", "wandb", "kubeflow", "bentoml",
    "data science", "reinforcement learning", "statistical modeling",
    "time series", "forecasting", "a/b testing",
    "gradient boosting", "random forest", "xgboost",
})

# Computer vision / speech — wrong domain for this JD
CV_SKILLS: frozenset[str] = frozenset({
    "yolo", "gans", "opencv", "asr", "image classification",
    "computer vision", "speech recognition", "cnn",
    "object detection", "diffusion models", "tts",
    "text to speech", "resnet", "vgg", "efficientnet",
    "generative adversarial", "stable diffusion",
})

# Career description evidence keywords (used with str.count for signal strength).
# IMPORTANT: Use specific JD-signal terms only — NOT generic tech words.
# Generic words like "search", "model", "production", "ranking", "query" appear
# in virtually every tech professional's career description and inflate scores
# for non-technical candidates (the keyword-stuffing trap the JD warns about).
_RETRIEVAL_EVIDENCE_KEYWORDS: tuple[str, ...] = (
    "information retrieval",
    "semantic search",
    "vector search",
    "hybrid search",
    "dense retrieval",
    "sparse retrieval",
    "bm25",
    "elasticsearch",
    "opensearch",
    "retrieval system",
    "search relevance",
    "search ranking",
    "inverted index",
    "lexical search",
    "retrieval pipeline",
)

_RANKING_EVIDENCE_KEYWORDS: tuple[str, ...] = (
    "learning to rank",
    "rerank",
    "re-rank",
    "ndcg",
    "mrr",
    "ranking system",
    "pointwise",
    "pairwise",
    "listwise",
    "rank fusion",
    "reciprocal rank",
    "mean average precision",
    "ltr model",
    "relevance model",
)

_RECOMMENDATION_EVIDENCE_KEYWORDS: tuple[str, ...] = (
    "recommendation system",
    "recommendation engine",
    "recommender",
    "collaborative filtering",
    "matrix factorization",
    "two-tower",
    "feed ranking",
    "content ranking",
    "candidate retrieval",
    "item embedding",
    "user embedding",
)

_VECTOR_DB_EVIDENCE_KEYWORDS: tuple[str, ...] = (
    "pinecone",
    "qdrant",
    "milvus",
    "weaviate",
    "faiss",
    "vector database",
    "vector store",
    "vector index",
    "pgvector",
    "embedding index",
    "approximate nearest neighbor",
    "chroma",
    "vespa",
    "opensearch knn",
)

_ML_PRODUCTION_EVIDENCE_KEYWORDS: tuple[str, ...] = (
    "shipped",
    "model in production",
    "inference pipeline",
    "model latency",
    "model throughput",
    "model serving",
    "millions of queries",
    "real users",
    "a/b test",
    "online evaluation",
    "offline evaluation",
    "ml pipeline",
)


# ---------------------------------------------------------------------------
# 3.  Title classification helpers
# ---------------------------------------------------------------------------

def classify_title(title: str) -> TitleCategory:
    """
    Map a free-text job title to a `TitleCategory`.

    Uses substring matching on lowercased title; more specific patterns
    are checked before broader ones to avoid false positives
    (e.g., 'AI Research Engineer' should be AI_ENGINEER, not SOFTWARE_ENGINEER).

    Parameters
    ----------
    title : str
        Raw `current_title` string from the candidate profile.

    Returns
    -------
    TitleCategory
    """
    t = title.lower().strip()

    # ── AI / ML (most specific first) ───────────────────────────────────────
    if any(kw in t for kw in (
        "ai engineer", "ai/ml", "applied scientist", "applied ml",
        "nlp engineer", "search engineer", "recommendation",
        "ranking engineer", "retrieval engineer",
    )):
        return TitleCategory.AI_ENGINEER

    if any(kw in t for kw in (
        "machine learning engineer", "ml engineer", "ml researcher",
        "senior ml", "staff ml", "principal ml", "junior ml",
        "computer vision engineer",  # CV is ML-adjacent
    )):
        return TitleCategory.ML_ENGINEER

    if any(kw in t for kw in (
        "data scientist", "applied scientist", "research scientist",
        "ai researcher", "ai research",
    )):
        return TitleCategory.DATA_SCIENTIST

    if any(kw in t for kw in (
        "data engineer", "analytics engineer", "etl engineer",
        "data platform", "pipeline engineer",
    )):
        return TitleCategory.DATA_ENGINEER

    # ── Software engineering ─────────────────────────────────────────────────
    if any(kw in t for kw in (
        "software engineer", "backend engineer", "frontend engineer",
        "fullstack", "full stack", "full-stack",
        "developer", "sde ", "sde1", "sde2", "sde3",
        "cloud engineer", "platform engineer", "infrastructure engineer",
        "mobile developer", "java developer", ".net developer",
        "qa engineer", "test engineer",
    )):
        return TitleCategory.SOFTWARE_ENGINEER

    # ── DevOps / SRE ─────────────────────────────────────────────────────────
    if any(kw in t for kw in (
        "devops", "sre", "site reliability", "devsecops", "mlops",
        "platform", "infrastructure",
    )):
        return TitleCategory.DEVOPS

    # ── Explicitly non-technical (checked BEFORE generic manager branch) ──────
    # Important: "marketing manager", "hr manager" etc. contain the word "manager"
    # so non-technical patterns must be matched first with higher specificity.
    _NON_TECH_PATTERNS: tuple[str, ...] = (
        "marketing", "accountant", "hr manager", "human resources",
        "customer support", "content writer", "graphic designer",
        "civil engineer", "mechanical engineer", "teacher", "lawyer",
        "operations manager", "business analyst", "project manager",
        "finance manager", "brand manager", "sales executive",
        "sales manager", "hr ", "finance ",
    )
    if any(kw in t for kw in _NON_TECH_PATTERNS):
        return TitleCategory.NON_TECHNICAL

    # ── Management / leadership (generic — after non-technical filter) ────────
    if any(kw in t for kw in (
        "engineering manager", "product manager", "director",
        "vp ", "vice president", "head of",
        "chief", "cto", "cpo", "coo", "lead ", "architect",
    )):
        return TitleCategory.MANAGER

    return TitleCategory.UNKNOWN


def _technical_title_score(title: str) -> float:
    """
    Continuous score 0.0–1.0 measuring how technical/relevant the title is.

    Used as a soft signal rather than hard filtering.
    """
    cat = classify_title(title)
    return {
        TitleCategory.AI_ENGINEER:       1.00,
        TitleCategory.ML_ENGINEER:       0.95,
        TitleCategory.DATA_SCIENTIST:    0.75,
        TitleCategory.DATA_ENGINEER:     0.55,
        TitleCategory.SOFTWARE_ENGINEER: 0.50,
        TitleCategory.DEVOPS:            0.25,
        TitleCategory.MANAGER:           0.20,
        TitleCategory.NON_TECHNICAL:     0.00,
        TitleCategory.UNKNOWN:           0.10,
    }[cat]


def _leadership_score(title: str) -> float:
    """
    Detect seniority/leadership indicators in the title.

    Staff/Principal > Senior > Lead > (default).
    Pure management without individual contribution is slightly negative here
    because the JD says 'this role writes code'.
    """
    t = title.lower()
    if any(kw in t for kw in ("staff", "principal", "distinguished")):
        return 0.90
    if any(kw in t for kw in ("senior", "sr.")):
        return 0.70
    if any(kw in t for kw in ("lead", "tech lead", "founding")):
        return 0.60
    if any(kw in t for kw in ("junior", "jr.", "associate", "intern")):
        return 0.20
    return 0.40   # default mid-level


# ---------------------------------------------------------------------------
# 4.  FeatureVector dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """
    Flat, primitive-typed feature representation of one candidate.

    All fields used by the scoring engine are here.
    The scoring engine should only read from this object — never from the
    raw CandidateRecord — so all feature engineering is centralised here.

    Default values
    --------------
    Numeric fields default to 0 / 0.0 / False so the scoring engine can
    apply arithmetic without null-checking.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    candidate_id: str

    # ── Basic experience ─────────────────────────────────────────────────────
    years_experience:       float = 0.0
    total_roles:            int   = 0
    avg_tenure_months:      float = 0.0

    # ── Title features ────────────────────────────────────────────────────────
    current_title:          str   = ""
    title_category:         str   = TitleCategory.UNKNOWN.value
    technical_title_score:  float = 0.0
    leadership_score:       float = 0.0

    # ── Company features ─────────────────────────────────────────────────────
    product_company_count:  int   = 0
    service_company_count:  int   = 0
    product_company_ratio:  float = 0.0

    # ── Skill counts by domain ────────────────────────────────────────────────
    retrieval_skill_count:      int = 0
    vector_db_skill_count:      int = 0
    llm_skill_count:            int = 0
    ranking_skill_count:        int = 0
    recommendation_skill_count: int = 0
    ml_skill_count:             int = 0
    cv_skill_count:             int = 0

    # ── Skill quality ─────────────────────────────────────────────────────────
    advanced_skill_count:   int   = 0
    expert_skill_count:     int   = 0
    avg_skill_duration:     float = 0.0

    # ── Behavioral signals ────────────────────────────────────────────────────
    recruiter_response_rate:    float = 0.0
    avg_response_time_hours:    float = 168.0   # default: 1 week (penalised)
    github_activity_score:      float = -1.0    # -1 = not linked
    open_to_work:               bool  = False
    notice_period_days:         int   = 90      # default: median
    interview_completion_rate:  float = 0.5
    offer_acceptance_rate:      float = -1.0    # -1 = no history

    # ── Validation ────────────────────────────────────────────────────────────
    has_skill_assessments:      bool  = False
    average_skill_assessment:   float = 0.0

    # ── Career evidence counts (keyword mentions in career text) ──────────────
    retrieval_evidence_count:       int = 0
    ranking_evidence_count:         int = 0
    recommendation_evidence_count:  int = 0
    vector_db_evidence_count:       int = 0
    ml_production_evidence_count:   int = 0

    # ── Career quality ────────────────────────────────────────────────────────
    career_text_length:         int   = 0
    product_ml_experience_years: float = 0.0   # KEY SIGNAL: ML+product company YOE
    technical_role_ratio:       float = 0.0

    # ── Honeypot signals ──────────────────────────────────────────────────────
    honeypot_confidence:    float = 0.0
    honeypot_flag_count:    int   = 0

    # ── Activity recency (Change 4: last_active_date signal) ─────────────────
    days_since_active:      int        = 0     # days since last_active_date; 0 = not extracted

    # ── Trust layer (soft profile-authenticity multiplier) ────────────────────
    # trust_score: 0.70–1.00 multiplicative factor applied after honeypot decay.
    # Penalises borderline inconsistencies (salary vs YOE, mild skill inflation)
    # that do not rise to the honeypot threshold.
    trust_score:            float = 1.0

    # ── Verification layer (platform-verified signals) ────────────────────────
    # verification_score: 0–100. Composite of Redrob-verified contact info,
    # LinkedIn connection, profile completeness, and GitHub linkage.
    # Only uses fields that Redrob verifies — NOT self-reported data.
    verification_score:     float = 0.0
    profile_completeness:   float = 0.0   # raw profile_completeness_score (0-100)

    # ── Explainability helpers (for factual reasoning text) ───────────────────
    top_domain_skills:      list[str]  = field(default_factory=list)  # top 3 JD-relevant skill names


# ---------------------------------------------------------------------------
# 5.  FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Converts a CandidateRecord + HoneypotResult → FeatureVector.

    Stateless — safe to reuse across the entire 100K candidate stream.

    Usage
    -----
    ```python
    extractor = FeatureExtractor()
    features  = extractor.extract(record, honeypot_result)
    ```
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self._config = config
        self._career_cfg: CareerConfig = config.career

    # ── Public interface ─────────────────────────────────────────────────────

    def extract(
        self,
        candidate: CandidateRecord,
        honeypot_result: Optional[HoneypotResult] = None,
    ) -> FeatureVector:
        """
        Extract all features from `candidate` into a flat `FeatureVector`.

        Parameters
        ----------
        candidate : CandidateRecord
            Parsed candidate (output of the parser layer).
        honeypot_result : HoneypotResult | None
            Pre-computed honeypot result.  If None, honeypot fields are 0.0.

        Returns
        -------
        FeatureVector
            Flat feature representation ready for the scoring engine.
        """
        fv = FeatureVector(candidate_id=candidate.candidate_id)

        # Run all extraction groups
        self._extract_experience(fv, candidate)
        self._extract_title(fv, candidate)
        self._extract_companies(fv, candidate)
        self._extract_skills(fv, candidate)
        self._extract_behavioral(fv, candidate)
        self._extract_assessments(fv, candidate)
        self._extract_career_evidence(fv, candidate)
        self._extract_career_quality(fv, candidate)
        self._extract_honeypot(fv, honeypot_result)
        self._extract_trust(fv, candidate)
        self._extract_verification(fv, candidate)

        logger.debug(
            "Extracted features for %s: title_cat=%s, prod_ml_yoe=%.1f, "
            "retrieval_ev=%d, honeypot_conf=%.2f",
            candidate.candidate_id,
            fv.title_category,
            fv.product_ml_experience_years,
            fv.retrieval_evidence_count,
            fv.honeypot_confidence,
        )
        return fv

    def extract_batch(
        self,
        candidates: list[CandidateRecord],
        honeypot_results: Optional[dict[str, HoneypotResult]] = None,
        *,
        log_every: int = 10_000,
    ) -> list[FeatureVector]:
        """
        Extract features for a list of candidates.

        Parameters
        ----------
        candidates : list[CandidateRecord]
        honeypot_results : dict[str, HoneypotResult] | None
            Mapping candidate_id → HoneypotResult (from HoneypotDetector.detect_batch).
            If None, honeypot features are zeroed.
        log_every : int
            Progress log interval.

        Returns
        -------
        list[FeatureVector]
            Parallel list — same order as `candidates`.
        """
        results: list[FeatureVector] = []
        hp_map = honeypot_results or {}

        for i, candidate in enumerate(candidates, start=1):
            hp = hp_map.get(candidate.candidate_id)
            fv = self.extract(candidate, hp)
            results.append(fv)
            if i % log_every == 0:
                logger.info("Feature extraction: %d/%d candidates", i, len(candidates))

        logger.info("Feature extraction complete: %d candidates", len(results))
        return results

    # ── Private extraction methods ───────────────────────────────────────────

    def _extract_experience(
        self, fv: FeatureVector, candidate: CandidateRecord
    ) -> None:
        """Basic experience fields."""
        fv.years_experience = candidate.total_yoe
        fv.total_roles = len(candidate.career_history)

        if candidate.career_history:
            total_months = sum(r.duration_months for r in candidate.career_history)
            fv.avg_tenure_months = total_months / len(candidate.career_history)
        else:
            fv.avg_tenure_months = 0.0

    def _extract_title(
        self, fv: FeatureVector, candidate: CandidateRecord
    ) -> None:
        """Title classification and scoring."""
        title = candidate.profile.current_title
        cat   = classify_title(title)

        fv.current_title         = title
        fv.title_category        = cat.value
        fv.technical_title_score = _technical_title_score(title)
        fv.leadership_score      = _leadership_score(title)

    def _extract_companies(
        self, fv: FeatureVector, candidate: CandidateRecord
    ) -> None:
        """
        Product vs services company distribution across career history.

        Matching strategy (in priority order):
        1. Company name matches known product companies list.
        2. Company name matches known services companies list.
        3. Industry string matches product/services industry sets.
        4. Falls through to 'neither' (neither counted).
        """
        product_cfg  = self._career_cfg.product_companies
        services_cfg = self._career_cfg.services_companies
        prod_ind     = self._career_cfg.product_industries
        svc_ind      = self._career_cfg.services_industries

        prod_count = 0
        svc_count  = 0

        for role in candidate.career_history:
            co_lower  = role.company.lower()
            ind_lower = role.industry.lower()

            is_product = (
                any(pk in co_lower  for pk in product_cfg)
                or any(pk in ind_lower for pk in prod_ind)
            )
            is_service = (
                any(sk in co_lower  for sk in services_cfg)
                or any(sk in ind_lower for sk in svc_ind)
            )

            if is_product and not is_service:
                prod_count += 1
            elif is_service:
                svc_count += 1

        total = len(candidate.career_history)
        fv.product_company_count = prod_count
        fv.service_company_count = svc_count
        fv.product_company_ratio = prod_count / total if total > 0 else 0.0

    def _extract_skills(
        self, fv: FeatureVector, candidate: CandidateRecord
    ) -> None:
        """
        Skill domain counts and quality metrics.

        Domain membership uses lowercase substring matching against the
        module-level frozensets.  A skill can belong to multiple domains
        (e.g., FAISS is both retrieval and vector_db).
        """
        retrieval_count      = 0
        vector_db_count      = 0
        llm_count            = 0
        ranking_count        = 0
        recommendation_count = 0
        ml_count             = 0
        cv_count             = 0
        advanced_count       = 0
        expert_count         = 0
        total_duration       = 0
        skill_count          = 0

        for sk in candidate.skills:
            sk_lower = sk.name.lower()

            # Domain membership
            if any(kw in sk_lower for kw in RETRIEVAL_SKILLS):
                retrieval_count += 1
            if any(kw in sk_lower for kw in VECTOR_DB_SKILLS):
                vector_db_count += 1
            if any(kw in sk_lower for kw in LLM_SKILLS):
                llm_count += 1
            if any(kw in sk_lower for kw in RANKING_SKILLS):
                ranking_count += 1
            if any(kw in sk_lower for kw in RECOMMENDATION_SKILLS):
                recommendation_count += 1
            if any(kw in sk_lower for kw in ML_SKILLS):
                ml_count += 1
            if any(kw in sk_lower for kw in CV_SKILLS):
                cv_count += 1

            # Proficiency quality
            if sk.proficiency == SkillProficiency.EXPERT:
                expert_count += 1
            elif sk.proficiency == SkillProficiency.ADVANCED:
                advanced_count += 1

            total_duration += sk.duration_months
            skill_count    += 1

        fv.retrieval_skill_count      = retrieval_count
        fv.vector_db_skill_count      = vector_db_count
        fv.llm_skill_count            = llm_count
        fv.ranking_skill_count        = ranking_count
        fv.recommendation_skill_count = recommendation_count
        fv.ml_skill_count             = ml_count
        fv.cv_skill_count             = cv_count
        fv.advanced_skill_count       = advanced_count
        fv.expert_skill_count         = expert_count
        fv.avg_skill_duration         = (total_duration / skill_count) if skill_count else 0.0

        # Collect top JD-relevant skill names for factual reasoning text.
        # Sorted by proficiency (expert first) so the most credible ones surface.
        _DOMAIN_SETS = (
            RETRIEVAL_SKILLS, VECTOR_DB_SKILLS,
            RANKING_SKILLS, RECOMMENDATION_SKILLS,
        )
        domain_skill_names: list[tuple[str, int]] = []
        for sk in candidate.skills:
            sk_lower = sk.name.lower()
            if any(any(kw in sk_lower for kw in ds) for ds in _DOMAIN_SETS):
                domain_skill_names.append((sk.name, sk.proficiency.numeric))
        domain_skill_names.sort(key=lambda x: -x[1])   # highest proficiency first
        fv.top_domain_skills = [name for name, _ in domain_skill_names[:3]]

    def _extract_behavioral(
        self, fv: FeatureVector, candidate: CandidateRecord
    ) -> None:
        """All 23 behavioral signals mapped to feature fields."""
        s = candidate.signals
        fv.recruiter_response_rate   = s.recruiter_response_rate
        fv.avg_response_time_hours   = s.avg_response_time_hours
        fv.github_activity_score     = s.github_activity_score
        fv.open_to_work              = s.open_to_work_flag
        fv.notice_period_days        = s.notice_period_days
        fv.interview_completion_rate = s.interview_completion_rate
        fv.offer_acceptance_rate     = s.offer_acceptance_rate
        # Change 4: activity recency — JD explicitly calls out that inactive
        # candidates are "not actually available for hiring purposes".
        fv.days_since_active         = candidate.days_since_active

    def _extract_assessments(
        self, fv: FeatureVector, candidate: CandidateRecord
    ) -> None:
        """Validated skill assessment scores."""
        assessments = candidate.signals.skill_assessment_scores
        if assessments:
            fv.has_skill_assessments    = True
            fv.average_skill_assessment = sum(assessments.values()) / len(assessments)
        else:
            fv.has_skill_assessments    = False
            fv.average_skill_assessment = 0.0

    def _extract_career_evidence(
        self, fv: FeatureVector, candidate: CandidateRecord
    ) -> None:
        """
        Count keyword occurrences in concatenated career descriptions.

        Uses str.count() so multiple mentions of 'search' or 'ranking' in a
        long career history accumulate — a stronger signal than a single mention.

        The career_text is pre-built by CandidateRecord.__post_init__() so
        this method does not re-iterate career history.
        """
        text = candidate.career_text.lower()

        fv.retrieval_evidence_count = sum(
            text.count(kw) for kw in _RETRIEVAL_EVIDENCE_KEYWORDS
        )
        fv.ranking_evidence_count = sum(
            text.count(kw) for kw in _RANKING_EVIDENCE_KEYWORDS
        )
        fv.recommendation_evidence_count = sum(
            text.count(kw) for kw in _RECOMMENDATION_EVIDENCE_KEYWORDS
        )
        fv.vector_db_evidence_count = sum(
            text.count(kw) for kw in _VECTOR_DB_EVIDENCE_KEYWORDS
        )
        fv.ml_production_evidence_count = sum(
            text.count(kw) for kw in _ML_PRODUCTION_EVIDENCE_KEYWORDS
        )
        fv.career_text_length = len(candidate.career_text)

    def _extract_career_quality(
        self, fv: FeatureVector, candidate: CandidateRecord
    ) -> None:
        """
        Derived career quality signals.

        product_ml_experience_years
        ---------------------------
        The single most important derived feature.  Sums months from roles that:
        - Are at a product company (by company name or industry), AND
        - Have a title OR description indicating ML/AI work.

        Detects the JD's "4-5 years applied ML at product companies" profile
        even when the candidate doesn't use buzzwords like "RAG" or "Pinecone".

        technical_role_ratio
        --------------------
        Fraction of all roles that appear technical (engineering/data/ML).
        Helps penalise candidates whose entire career history is non-technical
        even if the current title is technical.
        """
        product_cfg = self._career_cfg.product_companies
        prod_ind    = self._career_cfg.product_industries
        ml_keywords = self._career_cfg.ml_title_keywords

        prod_ml_months = 0.0
        tech_role_count = 0

        _TECH_ROLE_TITLE_WORDS: frozenset[str] = frozenset({
            "engineer", "developer", "scientist", "analyst", "architect",
            "researcher", "data", "ml", "ai", "sre", "devops", "platform",
        })

        # Change 1: Split ML evidence into specific (high-signal) and general (low-signal).
        # Problem: Generic words like "model", "search", "ranking", "neural" appear
        # in Business Analyst descriptions of the company's systems — NOT their own work.
        # Fix: Require 2+ *specific* ML terms, OR 5+ general terms with a clean title.

        # Specific: technologies/concepts that only appear in ML practitioners' descriptions
        _ML_DESC_EVIDENCE_STRONG: tuple[str, ...] = (
            "embedding", "vector search", "semantic search", "retrieval system",
            "recommendation system", "faiss", "pinecone", "weaviate", "milvus",
            "qdrant", "transformer", "sentence transformer", "learning to rank",
            "neural ranking", "ndcg", "vector database", "vector store",
        )
        # General: meaningful but appear in adjacent roles too
        _ML_DESC_EVIDENCE_GENERAL: tuple[str, ...] = (
            "machine learning", "neural network", "model training",
            "pytorch", "tensorflow", "scikit-learn", "nlp",
        )
        # Non-tech role patterns: these titles should NEVER accumulate ML-product
        # credit via description alone (they describe others' systems, not their own work)
        _NON_TECH_ROLE_PATTERNS: frozenset[str] = frozenset({
            "marketing", "hr manager", "human resources", "accountant",
            "customer support", "content writer", "graphic designer",
            "operations manager", "civil engineer", "mechanical engineer",
            "business analyst", "project manager", "sales",
        })

        for role in candidate.career_history:
            title_lower  = role.title.lower()
            desc_lower   = role.description.lower()
            co_lower     = role.company.lower()
            ind_lower    = role.industry.lower()

            # ── Is it a product company role? ────────────────────────────────
            at_product = (
                any(pk in co_lower  for pk in product_cfg)
                or any(pk in ind_lower for pk in prod_ind)
            )

            # ── Is it an ML/AI role? ──────────────────────────────────────────
            # ML title is the strongest signal (engineer, scientist, researcher)
            has_ml_title = any(kw in title_lower for kw in ml_keywords)

            # Strong description evidence: 2+ specific ML tech mentions
            # (FAISS, embedding, vector search — things a BA wouldn't mention)
            has_strong_ml_desc = (
                sum(desc_lower.count(kw) for kw in _ML_DESC_EVIDENCE_STRONG) >= 2
            )

            # General description evidence: only valid if the role title itself
            # is NOT a non-tech title (prevent BAs describing others' ML systems)
            role_is_non_tech = any(kw in title_lower for kw in _NON_TECH_ROLE_PATTERNS)
            has_general_ml_desc = (
                not role_is_non_tech
                and sum(desc_lower.count(kw) for kw in _ML_DESC_EVIDENCE_GENERAL) >= 5
            )

            has_ml_desc = has_strong_ml_desc or has_general_ml_desc

            if at_product and (has_ml_title or has_ml_desc):
                prod_ml_months += role.duration_months

            # ── Is it a technical role overall? ─────────────────────────────
            is_tech = any(kw in title_lower for kw in _TECH_ROLE_TITLE_WORDS)
            if is_tech:
                tech_role_count += 1

        total = len(candidate.career_history)
        fv.product_ml_experience_years = round(prod_ml_months / 12.0, 2)
        fv.technical_role_ratio        = tech_role_count / total if total > 0 else 0.0

    def _extract_honeypot(
        self,
        fv: FeatureVector,
        honeypot_result: Optional[HoneypotResult],
    ) -> None:
        """Pass-through of pre-computed honeypot signals."""
        if honeypot_result is None:
            fv.honeypot_confidence = 0.0
            fv.honeypot_flag_count  = 0
        else:
            fv.honeypot_confidence = honeypot_result.confidence
            fv.honeypot_flag_count  = len(honeypot_result.flags)

    def _extract_trust(
        self,
        fv: FeatureVector,
        candidate: CandidateRecord,
    ) -> None:
        """
        Compute a soft trust multiplier (0.70–1.00) capturing borderline
        profile-authenticity signals that do not rise to honeypot level.

        Three checks (additive penalties, capped at -0.30 total):

        1. Salary vs YOE sanity
           - Senior candidate (YOE >= 5) expecting min < 8 LPA:
             Suggests inflated experience claim (senior label, junior salary).
           - Junior candidate (YOE < 2) expecting max > 40 LPA:
             Suggests inflated salary expectation inconsistent with YOE.

        2. Borderline skill-duration inflation (4x–6x zone)
           The honeypot detector hard-flags >= 6x ratios.
           The 4x–6x zone is suspicious but could represent heavy parallelism.
           We apply a soft sliding penalty proportional to position in zone.

        3. Profile completeness (dataset median: 56.8 from dataset_profile.md)
           Very sparse profiles (< 40%) get a mild penalty because they lack
           verifiable information to corroborate skill claims.
        """
        trust = 1.0
        s = candidate.signals
        yoe = candidate.total_yoe

        # ── 1. Salary vs YOE sanity ──────────────────────────────────────────
        sal = s.expected_salary_range_inr_lpa
        if sal is not None:
            if sal.min_lpa > 0 and yoe >= 5.0 and sal.min_lpa < 8.0:
                # Senior by YOE but expecting junior salary → inflated YOE claim
                trust -= 0.10
            if sal.max_lpa > 0 and yoe < 2.0 and sal.max_lpa > 40.0:
                # Junior by YOE but expecting senior salary → inflated salary claim
                trust -= 0.08

        # ── 2. Borderline skill-duration inflation (4x–6x zone) ──────────────
        total_skill_months = sum(sk.duration_months for sk in candidate.skills)
        yoe_months = yoe * 12.0
        if yoe_months > 0:
            ratio = total_skill_months / yoe_months
            if 4.0 <= ratio < 6.0:
                # Sliding penalty: 4x → 0.0, 6x → -0.10
                trust -= ((ratio - 4.0) / 2.0) * 0.10

        # ── 3. Profile completeness ───────────────────────────────────────────
        completeness = s.profile_completeness_score
        if completeness < 40.0:
            trust -= 0.05   # Very sparse — fewer verifiable facts
        elif completeness > 80.0:
            trust += 0.02   # Well-maintained profile — small positive

        fv.trust_score = round(max(0.70, min(1.0, trust)), 4)

    def _extract_verification(
        self,
        fv: FeatureVector,
        candidate: CandidateRecord,
    ) -> None:
        """
        Compute a 0–100 verification score from Redrob platform-verified signals.

        These fields are NOT self-reported — they are verified by the Redrob
        platform directly:

        | Signal                     | Points |
        |----------------------------|--------|
        | verified_email             |  +25   |
        | verified_phone             |  +25   |
        | linkedin_connected         |  +20   |
        | profile_completeness (0-100→0-20) | +20 max |
        | github_activity_score linked |  +10  |
        | Total                      | 100    |

        The verification_score is used in the scoring engine's skill_quality
        component as a secondary quality booster (max +10 pts out of 100).
        The profile_completeness raw value is stored separately for audit.
        """
        score = 0.0
        s = candidate.signals

        if s.verified_email:
            score += 25.0
        if s.verified_phone:
            score += 25.0
        if s.linkedin_connected:
            score += 20.0

        # profile_completeness_score (0-100) → 0-20 points
        score += min((s.profile_completeness_score / 100.0) * 20.0, 20.0)

        # GitHub linked (any score != -1) signals a real public portfolio
        if s.github_activity_score != -1.0:
            score += 10.0

        fv.verification_score  = round(min(score, 100.0), 2)
        fv.profile_completeness = s.profile_completeness_score
