"""
config/settings.py
==================
Single source of truth for every tuneable constant in the pipeline.

Design rationale
----------------
- No magic numbers anywhere else in the codebase.
- Frozen dataclasses enforce immutability at runtime.
- Grouped by concern so they are easy to audit and tune independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"  # pre-computed embeddings etc.

CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
JD_FILE = DATA_DIR / "job_description.docx"
EMBEDDINGS_FILE = ARTIFACTS_DIR / "career_embeddings.npy"
CANDIDATE_IDS_FILE = ARTIFACTS_DIR / "candidate_ids.json"
SUBMISSION_FILE = OUTPUTS_DIR / "submission.csv"


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline switches."""

    top_k: int = 100                        # How many candidates to output
    log_level: str = "INFO"
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Offline embedding precomputation settings.

    We use all-MiniLM-L6-v2:
    - 384-dim vectors
    - ~80MB model weight
    - Encodes 100K × ~200 token docs in ~3-4 min on CPU
    - 100K × 384 × float32 ≈ 154 MB stored on disk
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256                   # tune for RAM vs speed
    max_seq_length: int = 512              # truncate long descriptions
    normalize_embeddings: bool = True      # cosine sim == dot product after this


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringWeights:
    """
    Master weight table.

    All weights must sum to 1.0.  The scoring engine enforces this at startup.
    Weights are intentionally documented with rationale so they are auditable.

    Tier 1 (~75%): semantic + career substance
    Tier 2 (~20%): behavioral availability signals
    Tier 3 (~5%):  secondary quality signals
    """

    # ── Tier 1: Semantic & career fit ──────────────────────────────────────
    semantic_fit:           float = 0.30   # cosine sim of career text vs JD
    product_company:        float = 0.15   # product vs services background
    applied_ml_yoe:         float = 0.15   # ML/AI years at product companies

    # ── Tier 2: Behavioral availability ────────────────────────────────────
    availability:           float = 0.15   # recency + open_to_work + response rate
    notice_period:          float = 0.10   # JD prefers <30 days; median is 90

    # ── Tier 3: Supporting signals ─────────────────────────────────────────
    github_activity:        float = 0.05   # "shipper" signal
    interview_reliability:  float = 0.03   # interview_completion_rate
    tenure_stability:       float = 0.02   # avg role duration; flags title-chasers
    skill_assessments:      float = 0.05   # validated scores (only 24% have these)

    def validate(self) -> None:
        """Raise ValueError if weights do not sum to 1.0."""
        total = sum(
            getattr(self, f.name) for f in self.__dataclass_fields__.values()
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ScoringWeights must sum to 1.0, got {total:.6f}"
            )


# ---------------------------------------------------------------------------
# Honeypot detection thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HoneypotConfig:
    """
    Thresholds for identifying impossible / synthetic profiles.

    Derived from dataset analysis:
    - Tier A skills (universal padding) have avg_duration ~16 months
    - Real ML engineers accumulate skills in parallel — 4× YOE is generous
    - Timeline mismatches >24 months indicate synthetic data errors
    """

    # Total skill duration / (YOE * 12) ratio above which we flag honeypot
    skill_duration_yoe_ratio: float = 6.0

    # Allowed delta (months) between stated duration_months and
    # calculated start-end difference before flagging a timeline mismatch
    timeline_mismatch_months: int = 24

    # Score assigned to confirmed honeypots (effectively removes them)
    honeypot_score: float = 0.0

    # Ratio threshold to flag (not hard-filter) borderline cases
    borderline_ratio: float = 4.0
    borderline_score_cap: float = 0.15


# ---------------------------------------------------------------------------
# Career substance scoring helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CareerConfig:
    """
    Configuration for career history feature extraction.

    Company lists come from dataset_profile.md analysis.
    Industry lists are normalised lowercase strings.
    """

    # Companies that signal product-company experience (high value)
    product_companies: frozenset[str] = field(default_factory=lambda: frozenset({
        "swiggy", "razorpay", "cred", "flipkart", "zomato", "phonepe",
        "freshworks", "zoho", "dream11", "inmobi", "meesho", "nykaa",
        "byjus", "byju's", "policybazaar", "ola", "vedantu", "paytm",
        "unacademy", "pharmeasy", "upgrad", "glance", "genpact ai",
        "pied piper", "hooli",   # fictional product cos in the dataset
    }))

    # Companies that signal IT-services / consulting (lower value)
    services_companies: frozenset[str] = field(default_factory=lambda: frozenset({
        "infosys", "wipro", "tcs", "hcl", "capgemini", "cognizant",
        "accenture", "mindtree", "tech mahindra", "mphasis",
        "l&t infotech", "ltimindtree", "hexaware", "niit technologies",
    }))

    # Industries that indicate product-company context
    product_industries: frozenset[str] = field(default_factory=lambda: frozenset({
        "software", "saas", "fintech", "edtech", "healthtech", "ecommerce",
        "e-commerce", "marketplace", "internet", "ai", "ml", "ai/ml",
        "food delivery", "gaming", "adtech", "transportation",
        "insurance tech", "healthtech ai", "conversational ai",
        "ai services",
    }))

    # Industries that signal services / consulting
    services_industries: frozenset[str] = field(default_factory=lambda: frozenset({
        "it services", "consulting", "bpo", "staffing", "outsourcing",
    }))

    # Titles that classify a role as ML/AI engineering
    ml_title_keywords: frozenset[str] = field(default_factory=lambda: frozenset({
        "machine learning", "ml engineer", "ai engineer", "nlp engineer",
        "data scientist", "applied ml", "search engineer",
        "recommendation", "ranking", "retrieval", "research engineer",
        "senior data scientist", "staff ml", "principal ml",
    }))

    # Title keywords that are hard disqualifiers (non-technical)
    disqualifying_title_keywords: frozenset[str] = field(default_factory=lambda: frozenset({
        "marketing manager", "hr manager", "accountant", "customer support",
        "sales executive", "content writer", "graphic designer",
        "operations manager", "civil engineer", "mechanical engineer",
        "project manager",
    }))

    # Experience band the JD targets (inclusive)
    target_yoe_min: float = 4.0
    target_yoe_max: float = 12.0

    # Min avg role tenure before flagging as title-chaser (months)
    min_avg_tenure_months: float = 18.0


# ---------------------------------------------------------------------------
# Availability / behavioral signal configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BehavioralConfig:
    """
    Thresholds and decay parameters for behavioral signals.

    Derived from signal stats:
    - Median notice_period_days = 90  (JD wants <30)
    - Median recruiter_response_rate = 0.44
    - Median avg_response_time_hours  = 129.9
    """

    # Days since last_active_date beyond which we start penalising
    activity_decay_start_days: int = 30
    # Days beyond which the candidate is treated as fully inactive
    activity_decay_full_days: int = 180

    # Notice period bands (days)
    notice_excellent: int = 30    # ideal — can buy out
    notice_good: int = 60         # manageable
    notice_poor: int = 90         # median; starts hurting score
    notice_bad: int = 120         # strong negative

    # Response rate below which we start penalising
    response_rate_threshold: float = 0.30

    # Response time above which we start penalising (hours)
    response_time_penalty_hours: float = 72.0


# ---------------------------------------------------------------------------
# JD embedding query
# ---------------------------------------------------------------------------

JD_QUERY_TEXT: str = (
    "Production experience building embeddings-based retrieval systems, "
    "vector databases, semantic search, ranking systems, recommendation engines. "
    "Shipped end-to-end ML systems to real users at product companies. "
    "Python, sentence-transformers, FAISS, Pinecone, Elasticsearch, hybrid search. "
    "Evaluation frameworks: NDCG, MRR, MAP. "
    "LLM fine-tuning, learning-to-rank, NLP, information retrieval. "
    "5 to 9 years applied ML experience at product companies, not consulting firms."
)


# ---------------------------------------------------------------------------
# Assembled config singleton
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Assembles all sub-configs into a single injectable object."""

    pipeline:    PipelineConfig    = field(default_factory=PipelineConfig)
    embedding:   EmbeddingConfig   = field(default_factory=EmbeddingConfig)
    weights:     ScoringWeights    = field(default_factory=ScoringWeights)
    honeypot:    HoneypotConfig    = field(default_factory=HoneypotConfig)
    career:      CareerConfig      = field(default_factory=CareerConfig)
    behavioral:  BehavioralConfig  = field(default_factory=BehavioralConfig)

    def validate(self) -> None:
        """Run all sub-config validations."""
        self.weights.validate()


# Default config instance — import this everywhere
DEFAULT_CONFIG = Config()
