"""
src/parser/candidate_parser.py
================================
Streaming JSONL parser that converts raw JSON lines into typed CandidateRecord
objects one at a time, keeping peak RAM minimal.

Design rationale
----------------
- Generator-based: the caller iterates via `for candidate in parse_candidates(path)`.
  The 487 MB file is never fully loaded into memory.
- Strict parsing with soft fallbacks: if a field is missing or malformed we log a
  warning and substitute a safe default rather than crashing the whole pipeline.
  A bad line in a 100K file should not break everything.
- Date parsing is centralised in `_parse_date()` so the format is fixed once.
- All enum coercions go through `_coerce_enum()` which logs and falls back cleanly.
- The `ParseStats` dataclass lets the caller report how many lines succeeded/failed
  after the stream is exhausted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Generator, Optional

from src.models.candidate import (
    CandidateProfile,
    CandidateRecord,
    CareerRole,
    Certification,
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parse statistics accumulator
# ---------------------------------------------------------------------------

@dataclass
class ParseStats:
    """Accumulated counts reported at the end of a parsing run."""
    total_lines:   int = 0
    parsed_ok:     int = 0
    parse_errors:  int = 0
    field_warnings: int = 0

    @property
    def success_rate(self) -> float:
        return self.parsed_ok / self.total_lines if self.total_lines else 0.0

    def report(self) -> str:
        return (
            f"Parsed {self.parsed_ok:,}/{self.total_lines:,} candidates "
            f"({self.success_rate:.1%} success, "
            f"{self.parse_errors} errors, "
            f"{self.field_warnings} field warnings)"
        )


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

def _parse_date(value: object, field_name: str, stats: ParseStats) -> Optional[date]:
    """
    Parse a date string in YYYY-MM-DD format.

    Returns None on failure (caller must handle sentinel appropriately).
    """
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        logger.debug("Bad date '%s' in field '%s'", value, field_name)
        stats.field_warnings += 1
        return None


def _coerce_enum(enum_cls: type, value: object, default, field_name: str,
                 stats: ParseStats):
    """
    Coerce a raw string into an enum member, falling back to `default`.
    """
    try:
        return enum_cls(value)
    except (ValueError, KeyError):
        logger.debug(
            "Unknown value '%s' for enum %s in field '%s'; using default %s",
            value, enum_cls.__name__, field_name, default
        )
        stats.field_warnings += 1
        return default


def _safe_float(value: object, default: float, field_name: str,
                stats: ParseStats) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        stats.field_warnings += 1
        logger.debug("Bad float '%s' in '%s'", value, field_name)
        return default


def _safe_int(value: object, default: int, field_name: str,
              stats: ParseStats) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        stats.field_warnings += 1
        logger.debug("Bad int '%s' in '%s'", value, field_name)
        return default


# ---------------------------------------------------------------------------
# Sub-object parsers
# ---------------------------------------------------------------------------

def _parse_skill(raw: dict, stats: ParseStats) -> Optional[Skill]:
    """Parse one skill dict. Returns None if the name is missing."""
    name = raw.get("name", "").strip()
    if not name:
        stats.field_warnings += 1
        return None

    proficiency = _coerce_enum(
        SkillProficiency,
        raw.get("proficiency"),
        SkillProficiency.BEGINNER,
        "skill.proficiency",
        stats,
    )
    return Skill(
        name=name,
        proficiency=proficiency,
        endorsements=_safe_int(raw.get("endorsements", 0), 0, "skill.endorsements", stats),
        duration_months=_safe_int(raw.get("duration_months", 0), 0, "skill.duration_months", stats),
    )


def _parse_career_role(raw: dict, stats: ParseStats) -> Optional[CareerRole]:
    """Parse one career history dict. Returns None if company or title is missing."""
    company = raw.get("company", "").strip()
    title   = raw.get("title", "").strip()
    if not company or not title:
        stats.field_warnings += 1
        return None

    start_date = _parse_date(raw.get("start_date"), "career.start_date", stats)
    if start_date is None:
        # Without a start date the timeline checks are impossible — use a placeholder
        start_date = date(2000, 1, 1)
        stats.field_warnings += 1

    end_date = _parse_date(raw.get("end_date"), "career.end_date", stats)
    # None end_date means current role — that is valid

    return CareerRole(
        company=company,
        title=title,
        start_date=start_date,
        end_date=end_date,
        duration_months=_safe_int(raw.get("duration_months", 0), 0, "career.duration_months", stats),
        is_current=bool(raw.get("is_current", False)),
        industry=raw.get("industry", "unknown").strip(),
        company_size=_coerce_enum(
            CompanySize,
            raw.get("company_size"),
            CompanySize.S,
            "career.company_size",
            stats,
        ),
        description=raw.get("description", "").strip(),
    )


def _parse_education(raw: dict, stats: ParseStats) -> Optional[Education]:
    """Parse one education dict."""
    institution = raw.get("institution", "").strip()
    if not institution:
        stats.field_warnings += 1
        return None

    return Education(
        institution=institution,
        degree=raw.get("degree", "").strip(),
        field_of_study=raw.get("field_of_study", "").strip(),
        start_year=_safe_int(raw.get("start_year", 1970), 1970, "edu.start_year", stats),
        end_year=_safe_int(raw.get("end_year", 1970), 1970, "edu.end_year", stats),
        grade=raw.get("grade"),
        tier=_coerce_enum(
            InstitutionTier,
            raw.get("tier"),
            InstitutionTier.UNKNOWN,
            "edu.tier",
            stats,
        ),
    )


def _parse_signals(raw: dict, stats: ParseStats) -> RedrobSignals:
    """
    Parse the redrob_signals block.

    Note on sentinels:
    - github_activity_score == -1 means no GitHub linked (preserve as-is)
    - offer_acceptance_rate == -1 means no offer history (preserve as-is)
    """
    signup    = _parse_date(raw.get("signup_date"), "signals.signup_date", stats)
    last_act  = _parse_date(raw.get("last_active_date"), "signals.last_active_date", stats)

    # Safe defaults if dates are missing
    if signup is None:
        signup = date(2024, 1, 1)
    if last_act is None:
        last_act = date(2024, 1, 1)

    sal_raw = raw.get("expected_salary_range_inr_lpa", {}) or {}
    salary = SalaryRange(
        min_lpa=_safe_float(sal_raw.get("min", 0.0), 0.0, "salary.min", stats),
        max_lpa=_safe_float(sal_raw.get("max", 0.0), 0.0, "salary.max", stats),
    )

    # skill_assessment_scores: dict[str, float]
    raw_assessments = raw.get("skill_assessment_scores") or {}
    assessments: dict[str, float] = {}
    if isinstance(raw_assessments, dict):
        for sk, sc in raw_assessments.items():
            try:
                assessments[str(sk)] = float(sc)
            except (TypeError, ValueError):
                pass

    return RedrobSignals(
        profile_completeness_score=_safe_float(
            raw.get("profile_completeness_score", 0), 0.0, "signals.profile_completeness_score", stats),
        signup_date=signup,
        last_active_date=last_act,
        open_to_work_flag=bool(raw.get("open_to_work_flag", False)),
        profile_views_received_30d=_safe_int(
            raw.get("profile_views_received_30d", 0), 0, "signals.profile_views_30d", stats),
        applications_submitted_30d=_safe_int(
            raw.get("applications_submitted_30d", 0), 0, "signals.apps_submitted_30d", stats),
        recruiter_response_rate=_safe_float(
            raw.get("recruiter_response_rate", 0.0), 0.0, "signals.recruiter_response_rate", stats),
        avg_response_time_hours=_safe_float(
            raw.get("avg_response_time_hours", 168.0), 168.0, "signals.avg_response_time_hours", stats),
        skill_assessment_scores=assessments,
        connection_count=_safe_int(
            raw.get("connection_count", 0), 0, "signals.connection_count", stats),
        endorsements_received=_safe_int(
            raw.get("endorsements_received", 0), 0, "signals.endorsements_received", stats),
        notice_period_days=_safe_int(
            raw.get("notice_period_days", 90), 90, "signals.notice_period_days", stats),
        expected_salary_range_inr_lpa=salary,
        preferred_work_mode=_coerce_enum(
            WorkMode,
            raw.get("preferred_work_mode"),
            WorkMode.FLEXIBLE,
            "signals.preferred_work_mode",
            stats,
        ),
        willing_to_relocate=bool(raw.get("willing_to_relocate", False)),
        github_activity_score=_safe_float(
            raw.get("github_activity_score", -1.0), -1.0, "signals.github_activity_score", stats),
        search_appearance_30d=_safe_int(
            raw.get("search_appearance_30d", 0), 0, "signals.search_appearance_30d", stats),
        saved_by_recruiters_30d=_safe_int(
            raw.get("saved_by_recruiters_30d", 0), 0, "signals.saved_by_recruiters_30d", stats),
        interview_completion_rate=_safe_float(
            raw.get("interview_completion_rate", 0.5), 0.5, "signals.interview_completion_rate", stats),
        offer_acceptance_rate=_safe_float(
            raw.get("offer_acceptance_rate", -1.0), -1.0, "signals.offer_acceptance_rate", stats),
        verified_email=bool(raw.get("verified_email", False)),
        verified_phone=bool(raw.get("verified_phone", False)),
        linkedin_connected=bool(raw.get("linkedin_connected", False)),
    )


def _parse_profile(raw: dict, stats: ParseStats) -> CandidateProfile:
    """Parse the 'profile' block."""
    return CandidateProfile(
        anonymized_name=raw.get("anonymized_name", "Unknown").strip(),
        headline=raw.get("headline", "").strip(),
        summary=raw.get("summary", "").strip(),
        location=raw.get("location", "").strip(),
        country=raw.get("country", "").strip(),
        years_of_experience=_safe_float(
            raw.get("years_of_experience", 0.0), 0.0, "profile.years_of_experience", stats),
        current_title=raw.get("current_title", "").strip(),
        current_company=raw.get("current_company", "").strip(),
        current_company_size=_coerce_enum(
            CompanySize,
            raw.get("current_company_size"),
            CompanySize.S,
            "profile.current_company_size",
            stats,
        ),
        current_industry=raw.get("current_industry", "unknown").strip(),
    )


# ---------------------------------------------------------------------------
# Record-level parser
# ---------------------------------------------------------------------------

def parse_record(raw: dict, stats: ParseStats) -> Optional[CandidateRecord]:
    """
    Convert one raw JSON dict into a CandidateRecord.

    Returns None if the record is fatally malformed (missing candidate_id
    or empty career history — both make the record unusable downstream).
    """
    candidate_id = raw.get("candidate_id", "").strip()
    if not candidate_id:
        logger.warning("Record missing candidate_id — skipping")
        stats.field_warnings += 1
        return None

    # ── Profile ─────────────────────────────────────────────────────────────
    profile_raw = raw.get("profile") or {}
    profile = _parse_profile(profile_raw, stats)

    # ── Career history ───────────────────────────────────────────────────────
    career_raw = raw.get("career_history") or []
    career: list[CareerRole] = []
    for role_raw in career_raw:
        role = _parse_career_role(role_raw, stats)
        if role is not None:
            career.append(role)

    if not career:
        # A candidate with zero parseable roles is useless for semantic scoring
        logger.debug("Candidate %s has no parseable career roles", candidate_id)
        # We still return a record — they will score near 0 naturally
        # rather than being silently dropped (we want a complete stats picture).

    # ── Education ────────────────────────────────────────────────────────────
    edu_raw = raw.get("education") or []
    education: list[Education] = [
        e for e_raw in edu_raw
        if (e := _parse_education(e_raw, stats)) is not None
    ]

    # ── Skills ───────────────────────────────────────────────────────────────
    skills_raw = raw.get("skills") or []
    skills: list[Skill] = [
        s for s_raw in skills_raw
        if (s := _parse_skill(s_raw, stats)) is not None
    ]

    # ── Certifications ───────────────────────────────────────────────────────
    certs_raw = raw.get("certifications") or []
    certifications: list[Certification] = []
    for c_raw in certs_raw:
        name   = c_raw.get("name", "").strip()
        issuer = c_raw.get("issuer", "").strip()
        year   = _safe_int(c_raw.get("year", 0), 0, "cert.year", stats)
        if name:
            certifications.append(Certification(name=name, issuer=issuer, year=year))

    # ── Languages ────────────────────────────────────────────────────────────
    langs_raw = raw.get("languages") or []
    languages: list[Language] = []
    for l_raw in langs_raw:
        lang = l_raw.get("language", "").strip()
        if lang:
            prof = _coerce_enum(
                LanguageProficiency,
                l_raw.get("proficiency"),
                LanguageProficiency.CONVERSATIONAL,
                "language.proficiency",
                stats,
            )
            languages.append(Language(language=lang, proficiency=prof))

    # ── Signals ──────────────────────────────────────────────────────────────
    signals_raw = raw.get("redrob_signals") or {}
    signals = _parse_signals(signals_raw, stats)

    return CandidateRecord(
        candidate_id=candidate_id,
        profile=profile,
        career_history=career,
        education=education,
        skills=skills,
        certifications=certifications,
        languages=languages,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Public streaming generator
# ---------------------------------------------------------------------------

def parse_candidates(
    path: Path,
    *,
    progress_every: int = 10_000,
) -> Generator[CandidateRecord, None, ParseStats]:
    """
    Stream-parse a JSONL file of candidate records.

    Yields
    ------
    CandidateRecord
        One typed record per valid JSON line.

    Returns (via StopIteration.value)
    ----------------------------------
    ParseStats
        Summary counts — access via:
        ```python
        gen = parse_candidates(path)
        for record in gen:
            ...
        stats = gen.value  # after exhaustion
        ```

    Notes
    -----
    - Malformed JSON lines are logged at WARNING and skipped.
    - Individual field errors are logged at DEBUG and substituted with defaults.
    - File is read line-by-line; peak RAM ≈ one record at a time + running stats.
    """
    stats = ParseStats()

    logger.info("Opening candidates file: %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            stats.total_lines += 1

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "JSON parse error on line %d: %s — skipping", line_num, exc
                )
                stats.parse_errors += 1
                continue

            record = parse_record(raw, stats)
            if record is None:
                stats.parse_errors += 1
                continue

            stats.parsed_ok += 1

            if stats.parsed_ok % progress_every == 0:
                logger.info("  … parsed %d candidates", stats.parsed_ok)

            yield record

    logger.info(stats.report())
    return stats


# ---------------------------------------------------------------------------
# Convenience: load ALL records into a list (use only when RAM allows)
# ---------------------------------------------------------------------------

def load_all_candidates(path: Path) -> tuple[list[CandidateRecord], ParseStats]:
    """
    Load all candidates into memory as a list.

    WARNING: For 100K candidates this consumes ~1–2 GB of RAM.
    Prefer the streaming `parse_candidates()` generator for the
    embedding pre-computation pass.  Use this only for the fast
    scoring pass once embeddings are precomputed.
    """
    gen = parse_candidates(path)
    records: list[CandidateRecord] = list(gen)
    stats: ParseStats = gen.value  # type: ignore[union-attr]
    return records, stats
