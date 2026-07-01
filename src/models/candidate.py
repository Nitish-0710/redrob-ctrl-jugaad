"""
src/models/candidate.py
=======================
Typed domain models mirroring the candidate_schema.json contract.

Design rationale
----------------
- Every field from the schema is represented as a typed dataclass field.
- `frozen=True` on leaf nodes (Skill, Education, CareerRole, etc.) prevents
  accidental mutation during feature extraction.
- The top-level CandidateRecord is NOT frozen so the scoring engine can
  attach computed scores without re-instantiating.
- Dates are stored as `datetime.date` objects — never raw strings — so the
  behavioral engine can do arithmetic directly.
- All Optional fields default to None, not sentinel strings or -1 (except
  github_activity_score which uses -1 per the schema spec).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations (mirrors schema enum values)
# ---------------------------------------------------------------------------

class SkillProficiency(str, Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"
    EXPERT       = "expert"

    @property
    def numeric(self) -> int:
        """Map proficiency to an ordinal for arithmetic comparisons."""
        return {
            self.BEGINNER:     1,
            self.INTERMEDIATE: 2,
            self.ADVANCED:     3,
            self.EXPERT:       4,
        }[self]


class LanguageProficiency(str, Enum):
    BASIC          = "basic"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL   = "professional"
    NATIVE         = "native"


class CompanySize(str, Enum):
    XS   = "1-10"
    S    = "11-50"
    M    = "51-200"
    ML   = "201-500"
    L    = "501-1000"
    XL   = "1001-5000"
    XXL  = "5001-10000"
    XXXL = "10001+"


class WorkMode(str, Enum):
    REMOTE   = "remote"
    HYBRID   = "hybrid"
    ONSITE   = "onsite"
    FLEXIBLE = "flexible"


class InstitutionTier(str, Enum):
    TIER_1  = "tier_1"
    TIER_2  = "tier_2"
    TIER_3  = "tier_3"
    TIER_4  = "tier_4"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Leaf-level models (frozen — never mutate after parsing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Skill:
    """One entry in the candidate's skills array."""
    name:             str
    proficiency:      SkillProficiency
    endorsements:     int
    duration_months:  int                  # months the candidate has used this skill


@dataclass(frozen=True)
class CareerRole:
    """One role in the candidate's career history."""
    company:          str
    title:            str
    start_date:       date
    end_date:         Optional[date]       # None means current role
    duration_months:  int
    is_current:       bool
    industry:         str
    company_size:     CompanySize
    description:      str                  # free-text: THE primary signal


@dataclass(frozen=True)
class Education:
    """One education entry."""
    institution:    str
    degree:         str
    field_of_study: str
    start_year:     int
    end_year:       int
    grade:          Optional[str]
    tier:           InstitutionTier


@dataclass(frozen=True)
class Certification:
    name:   str
    issuer: str
    year:   int


@dataclass(frozen=True)
class Language:
    language:    str
    proficiency: LanguageProficiency


@dataclass(frozen=True)
class SalaryRange:
    min_lpa: float
    max_lpa: float


# ---------------------------------------------------------------------------
# Behavioral signals (all 23, typed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RedrobSignals:
    """
    The 23 behavioral signals from the Redrob platform.

    Sentinel values preserved per schema spec:
    - github_activity_score = -1  → no GitHub linked
    - offer_acceptance_rate  = -1 → no prior offer history
    """

    profile_completeness_score:   float
    signup_date:                  date
    last_active_date:             date
    open_to_work_flag:            bool
    profile_views_received_30d:   int
    applications_submitted_30d:   int
    recruiter_response_rate:      float
    avg_response_time_hours:      float
    skill_assessment_scores:      dict[str, float]   # skill_name → 0-100
    connection_count:             int
    endorsements_received:        int
    notice_period_days:           int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode:          WorkMode
    willing_to_relocate:          bool
    github_activity_score:        float              # -1 = not linked
    search_appearance_30d:        int
    saved_by_recruiters_30d:      int
    interview_completion_rate:    float
    offer_acceptance_rate:        float              # -1 = no history
    verified_email:               bool
    verified_phone:               bool
    linkedin_connected:           bool


# ---------------------------------------------------------------------------
# Top-level candidate profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateProfile:
    """Static profile fields (parsed from 'profile' key in JSON)."""
    anonymized_name:      str
    headline:             str
    summary:              str
    location:             str
    country:              str
    years_of_experience:  float
    current_title:        str
    current_company:      str
    current_company_size: CompanySize
    current_industry:     str


# ---------------------------------------------------------------------------
# CandidateRecord — the primary unit of the pipeline
# ---------------------------------------------------------------------------

@dataclass
class CandidateRecord:
    """
    The single domain object that flows through every pipeline stage.

    Mutable fields
    --------------
    - `score`           : final composite score (set by ScoringEngine)
    - `component_scores`: dict of per-component scores (set by ScoringEngine)
    - `honeypot_flags`  : list of flag strings (set by HoneypotDetector)
    - `is_honeypot`     : bool (set by HoneypotDetector)
    - `reasoning`       : recruiter-style explanation (set by ExplainabilityEngine)
    - `rank`            : final rank 1–100 (set by RankingEngine)

    Immutable fields
    ----------------
    All parsed data from the JSONL is frozen after parsing.
    """

    # ── Core identity ───────────────────────────────────────────────────────
    candidate_id:    str
    profile:         CandidateProfile
    career_history:  list[CareerRole]
    education:       list[Education]
    skills:          list[Skill]
    certifications:  list[Certification]
    languages:       list[Language]
    signals:         RedrobSignals

    # ── Pipeline annotations (mutable — set by downstream stages) ───────────
    score:             float = 0.0
    component_scores:  dict[str, float] = field(default_factory=dict)
    honeypot_flags:    list[str]        = field(default_factory=list)
    is_honeypot:       bool             = False
    reasoning:         str              = ""
    rank:              Optional[int]    = None

    # ── Derived / cached fields (set by feature extractor) ─────────────────
    career_text:       str = ""   # concatenated career descriptions for embedding
    current_title_lower: str = "" # cached lowercase for fast comparisons

    def __post_init__(self) -> None:
        """Pre-compute cheap derived fields once at construction time."""
        object.__setattr__(
            self, "current_title_lower",
            self.profile.current_title.lower()
        ) if hasattr(self, "__dataclass_fields__") else None
        self.current_title_lower = self.profile.current_title.lower()
        self.career_text = self._build_career_text()

    def _build_career_text(self) -> str:
        """
        Concatenate career role descriptions into a single text block
        for embedding.

        Strategy: include title + company + description for each role.
        More recent roles are NOT weighted more here — that is handled
        in the scoring engine via position-weighted cosine similarity.
        """
        parts: list[str] = []
        for role in self.career_history:
            role_text = (
                f"{role.title} at {role.company} ({role.industry}): "
                f"{role.description}"
            )
            parts.append(role_text)
        return " | ".join(parts)

    @property
    def total_yoe(self) -> float:
        """Convenience accessor."""
        return self.profile.years_of_experience

    @property
    def has_github(self) -> bool:
        """True if the candidate has a GitHub account linked."""
        return self.signals.github_activity_score != -1

    @property
    def has_offer_history(self) -> bool:
        """True if the candidate has prior offer history."""
        return self.signals.offer_acceptance_rate != -1

    @property
    def days_since_active(self) -> int:
        """Days since last_active_date as of today."""
        return (date.today() - self.signals.last_active_date).days
