"""
src/data/parser.py
==================
Converts raw candidate JSON dicts → typed Python dataclass objects.

Design principles:
  - Every field has a default/Optional — never raises KeyError on partial data
  - Sentinel values preserved as-is (github_activity_score=-1, offer_acceptance_rate=-1)
  - No feature computation here — parsing only
  - All date strings kept as str (date arithmetic done in feature layer)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Any

# ══════════════════════════════════════════════════════════════════════════════
# ATOMIC SUB-STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Skill:
    """A single skill entry from the candidate's skills list."""
    name:             str
    proficiency:      str            # beginner | intermediate | advanced | expert
    endorsements:     int = 0
    duration_months:  int = 0        # months the candidate has practised this skill

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        return cls(
            name            = str(d.get("name", "")).strip(),
            proficiency     = str(d.get("proficiency", "beginner")).lower().strip(),
            endorsements    = int(d.get("endorsements", 0) or 0),
            duration_months = int(d.get("duration_months", 0) or 0),
        )


@dataclass
class JobEntry:
    """A single position in the career history."""
    company:         str
    title:           str
    start_date:      Optional[str]   # ISO 8601 date string, e.g. "2019-03-01"
    end_date:        Optional[str]   # None if is_current
    duration_months: int
    is_current:      bool
    industry:        str
    company_size:    str
    description:     str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobEntry":
        return cls(
            company         = str(d.get("company", "")).strip(),
            title           = str(d.get("title", "")).strip(),
            start_date      = d.get("start_date"),
            end_date        = d.get("end_date"),
            duration_months = int(d.get("duration_months", 0) or 0),
            is_current      = bool(d.get("is_current", False)),
            industry        = str(d.get("industry", "")).strip(),
            company_size    = str(d.get("company_size", "")).strip(),
            description     = str(d.get("description", "")).strip(),
        )


@dataclass
class EducationEntry:
    """A single education record."""
    institution:  str
    degree:       str
    field_of_study: str
    start_year:   Optional[int]
    end_year:     Optional[int]
    grade:        Optional[str]
    tier:         str              # tier_1 | tier_2 | tier_3 | tier_4 | unknown

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EducationEntry":
        return cls(
            institution   = str(d.get("institution", "")).strip(),
            degree        = str(d.get("degree", "")).strip(),
            field_of_study= str(d.get("field_of_study", "")).strip(),
            start_year    = _safe_int(d.get("start_year")),
            end_year      = _safe_int(d.get("end_year")),
            grade         = d.get("grade"),
            tier          = str(d.get("tier", "unknown")).strip(),
        )

    @property
    def duration_years(self) -> Optional[int]:
        if self.start_year and self.end_year:
            return self.end_year - self.start_year
        return None


@dataclass
class Certification:
    name:   str
    issuer: str
    year:   Optional[int]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Certification":
        return cls(
            name   = str(d.get("name", "")).strip(),
            issuer = str(d.get("issuer", "")).strip(),
            year   = _safe_int(d.get("year")),
        )


@dataclass
class Language:
    language:    str
    proficiency: str    # basic | conversational | professional | native

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Language":
        return cls(
            language    = str(d.get("language", "")).strip(),
            proficiency = str(d.get("proficiency", "basic")).lower().strip(),
        )


@dataclass
class SalaryRange:
    min_lpa: Optional[float]
    max_lpa: Optional[float]

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "SalaryRange":
        if not d:
            return cls(min_lpa=None, max_lpa=None)
        return cls(
            min_lpa = _safe_float(d.get("min")),
            max_lpa = _safe_float(d.get("max")),
        )

    @property
    def is_inverted(self) -> bool:
        """True if max < min — a known honeypot / data quality issue."""
        if self.min_lpa is not None and self.max_lpa is not None:
            return self.max_lpa < self.min_lpa
        return False

    @property
    def midpoint(self) -> Optional[float]:
        if self.min_lpa is not None and self.max_lpa is not None:
            if not self.is_inverted:
                return (self.min_lpa + self.max_lpa) / 2.0
        return self.min_lpa


@dataclass
class RedrobSignals:
    """Platform-level behavioral and engagement signals from Redrob."""
    profile_completeness_score:  float
    signup_date:                 Optional[str]
    last_active_date:            Optional[str]
    open_to_work_flag:           bool
    profile_views_received_30d:  int
    applications_submitted_30d:  int
    recruiter_response_rate:     float
    avg_response_time_hours:     float
    skill_assessment_scores:     Dict[str, float]   # skill_name → 0-100 score
    connection_count:            int
    endorsements_received:       int
    notice_period_days:          int
    expected_salary:             SalaryRange
    preferred_work_mode:         str    # remote | hybrid | onsite | flexible
    willing_to_relocate:         bool
    github_activity_score:       float  # 0-100 or -1 sentinel (no GitHub)
    search_appearance_30d:       int
    saved_by_recruiters_30d:     int
    interview_completion_rate:   float
    offer_acceptance_rate:       float  # 0-1 or -1 sentinel (no history)
    verified_email:              bool
    verified_phone:              bool
    linkedin_connected:          bool

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RedrobSignals":
        sal_raw = d.get("expected_salary_range_inr_lpa")
        return cls(
            profile_completeness_score  = float(d.get("profile_completeness_score", 0) or 0),
            signup_date                 = d.get("signup_date"),
            last_active_date            = d.get("last_active_date"),
            open_to_work_flag           = bool(d.get("open_to_work_flag", False)),
            profile_views_received_30d  = int(d.get("profile_views_received_30d", 0) or 0),
            applications_submitted_30d  = int(d.get("applications_submitted_30d", 0) or 0),
            recruiter_response_rate     = float(d.get("recruiter_response_rate", 0) or 0),
            avg_response_time_hours     = float(d.get("avg_response_time_hours", 0) or 0),
            skill_assessment_scores     = dict(d.get("skill_assessment_scores") or {}),
            connection_count            = int(d.get("connection_count", 0) or 0),
            endorsements_received       = int(d.get("endorsements_received", 0) or 0),
            notice_period_days          = int(d.get("notice_period_days", 60) or 60),
            expected_salary             = SalaryRange.from_dict(sal_raw),
            preferred_work_mode         = str(d.get("preferred_work_mode", "flexible")).strip(),
            willing_to_relocate         = bool(d.get("willing_to_relocate", False)),
            github_activity_score       = float(d.get("github_activity_score", -1) if d.get("github_activity_score") is not None else -1),
            search_appearance_30d       = int(d.get("search_appearance_30d", 0) or 0),
            saved_by_recruiters_30d     = int(d.get("saved_by_recruiters_30d", 0) or 0),
            interview_completion_rate   = float(d.get("interview_completion_rate", 0) or 0),
            offer_acceptance_rate       = float(d.get("offer_acceptance_rate", -1) if d.get("offer_acceptance_rate") is not None else -1),
            verified_email              = bool(d.get("verified_email", False)),
            verified_phone              = bool(d.get("verified_phone", False)),
            linkedin_connected          = bool(d.get("linkedin_connected", False)),
        )

    @property
    def has_github(self) -> bool:
        return self.github_activity_score >= 0

    @property
    def has_offer_history(self) -> bool:
        return self.offer_acceptance_rate >= 0

    @property
    def avg_assessment_score(self) -> Optional[float]:
        if self.skill_assessment_scores:
            return sum(self.skill_assessment_scores.values()) / len(self.skill_assessment_scores)
        return None

    @property
    def n_assessments(self) -> int:
        return len(self.skill_assessment_scores)


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL CANDIDATE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateProfile:
    """Profile-level fields (non-nested)."""
    anonymized_name:      str
    headline:             str
    summary:              str
    location:             str
    country:              str
    years_of_experience:  float
    current_title:        str
    current_company:      str
    current_company_size: str
    current_industry:     str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateProfile":
        return cls(
            anonymized_name      = str(d.get("anonymized_name", "")).strip(),
            headline             = str(d.get("headline", "")).strip(),
            summary              = str(d.get("summary", "")).strip(),
            location             = str(d.get("location", "")).strip(),
            country              = str(d.get("country", "")).strip(),
            years_of_experience  = float(d.get("years_of_experience", 0) or 0),
            current_title        = str(d.get("current_title", "")).strip(),
            current_company      = str(d.get("current_company", "")).strip(),
            current_company_size = str(d.get("current_company_size", "")).strip(),
            current_industry     = str(d.get("current_industry", "")).strip(),
        )


@dataclass
class Candidate:
    """
    Top-level structured representation of a single candidate.

    This is the primary data object passed through the entire pipeline:
      Loader → Parser → Validator → Features → Ranker → Submission
    """
    candidate_id:   str
    profile:        CandidateProfile
    career_history: List[JobEntry]
    education:      List[EducationEntry]
    skills:         List[Skill]
    certifications: List[Certification]
    languages:      List[Language]
    redrob_signals: RedrobSignals

    # ── Derived convenience properties ────────────────────────────────────────

    @property
    def total_career_months(self) -> int:
        return sum(j.duration_months for j in self.career_history)

    @property
    def implied_experience_years(self) -> float:
        return round(self.total_career_months / 12.0, 2)

    @property
    def experience_discrepancy(self) -> float:
        """Claimed YoE minus career-derived YoE. Should be near 0 for honest candidates."""
        return round(self.profile.years_of_experience - self.implied_experience_years, 2)

    @property
    def skill_names(self) -> List[str]:
        return [s.name for s in self.skills]

    @property
    def n_skills(self) -> int:
        return len(self.skills)

    @property
    def n_jobs(self) -> int:
        return len(self.career_history)

    @property
    def current_job(self) -> Optional[JobEntry]:
        for job in self.career_history:
            if job.is_current:
                return job
        return None

    @property
    def highest_edu_tier(self) -> str:
        tier_rank = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}
        if not self.education:
            return "none"
        return min(self.education, key=lambda e: tier_rank.get(e.tier, 5)).tier

    def __repr__(self) -> str:
        return (
            f"Candidate({self.candidate_id!r}, "
            f"title={self.profile.current_title!r}, "
            f"yoe={self.profile.years_of_experience}, "
            f"skills={self.n_skills})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PARSING ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_candidate(raw: Dict[str, Any]) -> Candidate:
    """
    Convert a raw JSON dict (one JSONL line) into a typed Candidate object.

    Args:
        raw: The deserialized JSON dict for one candidate.

    Returns:
        A fully structured Candidate dataclass instance.

    Raises:
        KeyError:   If 'candidate_id' is missing (non-recoverable).
        ValueError: If a required nested structure is completely absent.
    """
    return Candidate(
        candidate_id   = str(raw["candidate_id"]),
        profile        = CandidateProfile.from_dict(raw.get("profile", {})),
        career_history = [JobEntry.from_dict(j) for j in raw.get("career_history", [])],
        education      = [EducationEntry.from_dict(e) for e in raw.get("education", [])],
        skills         = [Skill.from_dict(s) for s in raw.get("skills", [])],
        certifications = [Certification.from_dict(c) for c in raw.get("certifications", [])],
        languages      = [Language.from_dict(l) for l in raw.get("languages", [])],
        redrob_signals = RedrobSignals.from_dict(raw.get("redrob_signals", {})),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None
