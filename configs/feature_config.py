"""
configs/feature_config.py
==========================
Central registry for all feature definitions.

This is the SINGLE SOURCE OF TRUTH for:
  - Which skills are considered "AI core"
  - Proficiency level weights
  - Scoring thresholds
  - Honeypot detection parameters
  - Field-level trust multipliers

Feature engineers modify this file; ranking/scoring code reads from it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# AI CORE SKILL TAXONOMY
# Source: JD analysis + dataset forensics (dataset_analysis.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

AI_CORE_SKILLS: FrozenSet[str] = frozenset({
    # Foundations
    "Python", "Machine Learning", "Deep Learning", "Statistics", "Mathematics",
    "Statistical Modeling", "Data Science", "Feature Engineering", "A/B Testing",

    # LLMs & Language
    "NLP", "LLMs", "Fine-tuning LLMs", "Transformers", "Hugging Face", "HuggingFace",
    "LoRA", "QLoRA", "PEFT", "Prompt Engineering", "LangChain", "LlamaIndex", "OpenAI API", "RAG",

    # Computer Vision
    "Computer Vision", "Image Classification", "Object Detection",
    "GANs", "Diffusion Models",

    # Audio
    "Speech Recognition", "TTS",

    # Frameworks
    "PyTorch", "TensorFlow", "Scikit-learn", "Keras", "JAX",

    # Infra / MLOps
    "MLOps", "Model Deployment", "ONNX", "DVC", "MLFlow", "Kubeflow",
    "Weights & Biases", "BentoML", "TorchServe", "Triton", "FastAPI",

    # Vector / Retrieval / Search
    "Vector Databases", "Milvus", "Pinecone", "Weaviate", "FAISS", "Qdrant", "pgvector",
    "Elasticsearch", "OpenSearch", "Solr", "Lucene", "BM25", "Hybrid Search",

    # Advanced ML / Ranking
    "Reinforcement Learning", "Learning to Rank", "LambdaMART", "XGBoost Ranker", "Recommender Systems",
})

# Skills adjacent to AI — worth partial credit but not core
AI_ADJACENT_SKILLS: FrozenSet[str] = frozenset({
    "SQL", "Spark", "Airflow", "Kafka", "Databricks",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes",
    "Apache Beam", "Apache Flink",
    "Python",  # also in core — Python is always credited
    "R",
})


# ══════════════════════════════════════════════════════════════════════════════
# PROFICIENCY LEVEL WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

PROFICIENCY_WEIGHTS: Dict[str, float] = {
    "expert":       4.0,
    "advanced":     3.0,
    "intermediate": 2.0,
    "beginner":     1.0,
}

# Discount factor when a skill has 0 endorsements (credibility penalty)
ZERO_ENDORSEMENT_DISCOUNT: float = 0.6

# Endorsement saturation cap — beyond this endorsements don't add more weight
ENDORSEMENT_CAP: int = 50


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Ideal experience range for "Senior AI Engineer" in years
EXPERIENCE_IDEAL_MIN: float = 5.0
EXPERIENCE_IDEAL_MAX: float = 12.0
EXPERIENCE_HARD_MIN:  float = 3.0   # floor below which candidates are junior

# Max tolerated gap between claimed YoE and career-derived YoE (years)
YOE_DISCREPANCY_TOLERANCE: float = 2.0
YOE_DISCREPANCY_HONEYPOT:  float = 5.0  # beyond this → honeypot flag


# ══════════════════════════════════════════════════════════════════════════════
# EDUCATION TIER WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

EDUCATION_TIER_WEIGHTS: Dict[str, float] = {
    "tier_1":  1.0,
    "tier_2":  0.75,
    "tier_3":  0.50,
    "tier_4":  0.30,
    "unknown": 0.20,
    "none":    0.10,
}

# Field of study relevance to AI Engineering
RELEVANT_FIELDS: FrozenSet[str] = frozenset({
    "Computer Science", "Artificial Intelligence", "Machine Learning",
    "Data Science", "Information Technology", "Electronics",
    "Electrical Engineering", "Mathematics", "Statistics",
    "Computer Engineering", "Software Engineering",
})


# ══════════════════════════════════════════════════════════════════════════════
# COMPANY SIZE PRESTIGE
# ══════════════════════════════════════════════════════════════════════════════

COMPANY_SIZE_WEIGHTS: Dict[str, float] = {
    "10001+":    1.0,
    "5001-10000": 0.9,
    "1001-5000": 0.8,
    "501-1000":  0.7,
    "201-500":   0.6,
    "51-200":    0.55,
    "11-50":     0.50,
    "1-10":      0.45,
}


# ══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL SIGNAL WEIGHTS
# (Used when building the composite behavioral score)
# ══════════════════════════════════════════════════════════════════════════════

BEHAVIORAL_WEIGHTS: Dict[str, float] = {
    "recruiter_response_rate":    0.35,
    "interview_completion_rate":  0.30,
    "offer_acceptance_rate":      0.15,   # -1 sentinel → treated as 0
    "github_activity_score":      0.20,   # -1 sentinel → treated as 0
}


# ══════════════════════════════════════════════════════════════════════════════
# HONEYPOT DETECTION THRESHOLDS
# Source: forensics in dataset_analysis.ipynb
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HoneypotConfig:
    """Immutable configuration for all honeypot detection rules."""

    # YoE inflation: claimed > career_months/12 + this threshold → INFLATED_YOE
    yoe_inflation_threshold_years: float = 3.0

    # Skill duration: skill_months > total_career_months + grace → SKILL_OVERFLOW
    skill_duration_grace_months: int = 12

    # Overlap: two jobs overlap more than this many months → OVERLAP_JOBS
    job_overlap_grace_months: int = 3

    # Expert skill with 0 endorsements AND 0 assessment → UNENDORSED_EXPERT
    expert_min_endorsements: int = 1

    # Assessment score above this with 0 endorsements → HIGH_SCORE_ZERO_ENDORSE
    assessment_zero_endorse_threshold: float = 85.0

    # Ghost profile: 0 visibility + applications above this → GHOST_PROFILE
    ghost_application_threshold: int = 5

    # Trust penalty per flag (multiplicative, floored at min_trust)
    trust_penalty_per_flag: float = 0.12
    min_trust_score: float = 0.25


HONEYPOT_CONFIG = HoneypotConfig()


# ══════════════════════════════════════════════════════════════════════════════
# SALARY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Realistic salary range for Senior AI Engineer in India (INR LPA)
SALARY_REASONABLE_MIN_LPA: float = 15.0
SALARY_REASONABLE_MAX_LPA: float = 120.0

# Flag if max < min (inverted salary — honeypot indicator)
SALARY_INVERSION_DETECTION: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL NORMALIZATION BOUNDS
# (For min-max normalization; based on dataset_analysis.ipynb percentiles)
# ══════════════════════════════════════════════════════════════════════════════

NORMALIZATION_BOUNDS: Dict[str, Tuple[float, float]] = {
    "years_of_experience":       (0.0,   50.0),
    "n_skills":                  (0.0,   30.0),
    "n_ai_skills":               (0.0,   20.0),
    "profile_completeness_score":(0.0,  100.0),
    "recruiter_response_rate":   (0.0,    1.0),
    "interview_completion_rate": (0.0,    1.0),
    "github_activity_score":     (0.0,  100.0),
    "connection_count":          (0.0, 1000.0),
    "endorsements_received":     (0.0,  500.0),
    "profile_views_30d":         (0.0,  500.0),
    "search_appearance_30d":     (0.0, 1000.0),
    "saved_by_recruiters_30d":   (0.0,   50.0),
    "avg_assessment_score":      (0.0,  100.0),
}


# ══════════════════════════════════════════════════════════════════════════════
# CAREER EVIDENCE KEYWORDS
# (Used to parse headlines, summaries, and job descriptions)
# ══════════════════════════════════════════════════════════════════════════════

CAREER_EVIDENCE_KEYWORDS: Dict[str, FrozenSet[str]] = {
    "retrieval": frozenset({"retrieval", "search", "bm25", "semantic search", "vector search", "dense retrieval", "information retrieval"}),
    "ranking": frozenset({"ranking", "rank", "learning to rank", "ltr", "ndcg", "re-ranking", "rerank"}),
    "recommendation": frozenset({"recommendation", "recommender", "collaborative filtering", "personalization"}),
    "search": frozenset({"search engine", "elasticsearch", "solr", "lucene"}),
    "relevance": frozenset({"relevance", "relevant", "query understanding"}),
    "personalization": frozenset({"personalization", "personalized", "user behavior"}),
    "evaluation": frozenset({"evaluation", "a/b testing", "offline evaluation", "metrics", "ndcg", "mrr", "precision@k"}),
    "machine_learning": frozenset({"machine learning", "ml", "deep learning", "dl", "predictive modeling"}),
    "nlp": frozenset({"nlp", "natural language processing", "text mining", "llm", "large language model"}),
    "production_ml": frozenset({"production", "scale", "latency", "throughput", "mlops", "deployment", "serving", "high performance"})
}
