"""
src/data/validators.py
=======================
Schema validation and honeypot detection for Candidate objects.

Three layers of checks:
  1. SCHEMA checks — required fields present, types correct, enum values valid
  2. CONSISTENCY checks — internal logic (dates, salary, durations)
  3. HONEYPOT checks — suspicious patterns identified in dataset forensics

Each check is independent. Results accumulate into a ValidationResult
so all issues are reported in one pass (no fail-fast).

Key findings from dataset_analysis.ipynb that inform these validators:
  - 18,865 salary inversions (18.9% of dataset)
  - 16,500 skill duration overflows (16.5%)
  - 7,496 date paradoxes (7.5%)
  - 25 inflated YoE, 24 unendorsed experts, 7 high-score-zero-endorse
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional

from configs.feature_config import HONEYPOT_CONFIG, HoneypotConfig
from configs.settings import get_logger
from src.data.parser import Candidate, JobEntry, EducationEntry, Skill, RedrobSignals

logger = get_logger(__name__)

VALID_PROFICIENCY    = {"beginner", "intermediate", "advanced", "expert"}
VALID_COMPANY_SIZES  = {"1-10","11-50","51-200","201-500","501-1000","1001-5000","5001-10000","10001+"}
VALID_WORK_MODES     = {"remote","hybrid","onsite","flexible"}
VALID_EDU_TIERS      = {"tier_1","tier_2","tier_3","tier_4","unknown"}
REFERENCE_DATE       = date(2026, 6, 23)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Accumulates all validation findings for a single candidate."""
    candidate_id:     str
    errors:           List[str] = field(default_factory=list)   # blocking issues
    warnings:         List[str] = field(default_factory=list)   # non-blocking
    honeypot_flags:   List[str] = field(default_factory=list)   # suspicious patterns
    trust_score:      float = 1.0                               # 0.0 – 1.0

    @property
    def is_valid(self) -> bool:
        """True if no blocking schema errors were found."""
        return len(self.errors) == 0

    @property
    def is_clean(self) -> bool:
        """True if no errors, warnings, OR honeypot flags."""
        return self.is_valid and len(self.warnings) == 0 and len(self.honeypot_flags) == 0

    @property
    def n_flags(self) -> int:
        return len(self.honeypot_flags)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_honeypot(self, flag_type: str, detail: str) -> None:
        self.honeypot_flags.append(f"{flag_type}: {detail}")

    def compute_trust_score(self, cfg: HoneypotConfig = HONEYPOT_CONFIG) -> float:
        """
        Derive a trust multiplier from the number of honeypot flags.
        Each flag reduces trust by cfg.trust_penalty_per_flag,
        floored at cfg.min_trust_score.
        """
        penalty = self.n_flags * cfg.trust_penalty_per_flag
        self.trust_score = max(cfg.min_trust_score, 1.0 - penalty)
        return self.trust_score

    def __repr__(self) -> str:
        return (
            f"ValidationResult({self.candidate_id!r}, "
            f"valid={self.is_valid}, flags={self.n_flags}, "
            f"trust={self.trust_score:.2f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MASTER VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def validate_candidate(
    candidate: Candidate,
    cfg: HoneypotConfig = HONEYPOT_CONFIG,
) -> ValidationResult:
    """
    Run all validation layers for a single Candidate.

    Layers (in order):
      1. Schema validation
      2. Consistency checks
      3. Honeypot detection

    Returns:
        ValidationResult with accumulated findings and a computed trust_score.
    """
    result = ValidationResult(candidate_id=candidate.candidate_id)

    _validate_schema(candidate, result)
    _validate_consistency(candidate, result)
    _detect_honeypot(candidate, result, cfg)

    result.compute_trust_score(cfg)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1: SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _validate_schema(c: Candidate, r: ValidationResult) -> None:
    """Check required fields and enum membership."""

    # candidate_id format
    import re
    if not re.match(r"^CAND_\d{7}$", c.candidate_id):
        r.add_error(f"Invalid candidate_id format: {c.candidate_id!r}")

    # Profile
    p = c.profile
    if not p.current_title:
        r.add_warning("profile.current_title is empty")
    if not p.location:
        r.add_warning("profile.location is empty")
    if p.years_of_experience < 0 or p.years_of_experience > 50:
        r.add_error(f"years_of_experience out of range: {p.years_of_experience}")

    # Career history
    if len(c.career_history) == 0:
        r.add_error("career_history is empty — required to have at least 1 entry")
    for i, job in enumerate(c.career_history):
        if job.company_size and job.company_size not in VALID_COMPANY_SIZES:
            r.add_warning(f"career_history[{i}].company_size invalid: {job.company_size!r}")
        if job.duration_months < 0:
            r.add_error(f"career_history[{i}].duration_months is negative: {job.duration_months}")

    # Skills
    for i, skill in enumerate(c.skills):
        if not skill.name:
            r.add_warning(f"skills[{i}].name is empty")
        if skill.proficiency not in VALID_PROFICIENCY:
            r.add_warning(f"skills[{i}].proficiency invalid: {skill.proficiency!r}")
        if skill.endorsements < 0:
            r.add_warning(f"skills[{i}].endorsements is negative")
        if skill.duration_months < 0:
            r.add_warning(f"skills[{i}].duration_months is negative")

    # Education
    for i, edu in enumerate(c.education):
        if edu.tier not in VALID_EDU_TIERS:
            r.add_warning(f"education[{i}].tier invalid: {edu.tier!r}")

    # Redrob signals
    rs = c.redrob_signals
    if not (0 <= rs.profile_completeness_score <= 100):
        r.add_error(f"profile_completeness_score out of range: {rs.profile_completeness_score}")
    if rs.preferred_work_mode not in VALID_WORK_MODES:
        r.add_warning(f"preferred_work_mode invalid: {rs.preferred_work_mode!r}")
    if not (-1 <= rs.github_activity_score <= 100):
        r.add_error(f"github_activity_score out of range: {rs.github_activity_score}")
    if not (-1 <= rs.offer_acceptance_rate <= 1):
        r.add_error(f"offer_acceptance_rate out of range: {rs.offer_acceptance_rate}")
    if not (0 <= rs.recruiter_response_rate <= 1):
        r.add_error(f"recruiter_response_rate out of range: {rs.recruiter_response_rate}")
    if not (0 <= rs.interview_completion_rate <= 1):
        r.add_error(f"interview_completion_rate out of range: {rs.interview_completion_rate}")
    if rs.notice_period_days < 0 or rs.notice_period_days > 180:
        r.add_warning(f"notice_period_days unusual: {rs.notice_period_days}")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: CONSISTENCY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _validate_consistency(c: Candidate, r: ValidationResult) -> None:
    """Check internal logical consistency of the candidate record."""
    rs = c.redrob_signals

    # Salary range sanity
    sal = rs.expected_salary
    if sal.min_lpa is not None and sal.max_lpa is not None:
        if sal.max_lpa < sal.min_lpa:
            r.add_warning(
                f"Salary inverted: min={sal.min_lpa} LPA > max={sal.max_lpa} LPA"
            )
        if sal.min_lpa < 0:
            r.add_warning(f"Salary min is negative: {sal.min_lpa}")

    # Date consistency: last_active >= signup_date
    signup = _parse_date(rs.signup_date)
    active = _parse_date(rs.last_active_date)
    if signup and active:
        if active < signup:
            r.add_warning(
                f"last_active_date ({active}) is BEFORE signup_date ({signup})"
            )
        if active > REFERENCE_DATE:
            r.add_warning(
                f"last_active_date ({active}) is in the future (ref={REFERENCE_DATE})"
            )

    # Title match: profile.current_title should match career is_current title
    profile_title = c.profile.current_title.lower().strip()
    current_jobs  = [j for j in c.career_history if j.is_current]
    if current_jobs:
        career_titles = [j.title.lower().strip() for j in current_jobs]
        if profile_title and profile_title not in career_titles:
            r.add_warning(
                f"Profile title {profile_title!r} doesn't match "
                f"career current title(s): {career_titles}"
            )

    # At most one current job
    n_current = sum(1 for j in c.career_history if j.is_current)
    if n_current > 1:
        r.add_warning(f"Multiple 'is_current=True' jobs: {n_current}")

    # Education duration sanity
    for i, edu in enumerate(c.education):
        if edu.duration_years is not None:
            if edu.duration_years < 0:
                r.add_warning(
                    f"education[{i}] has negative duration: "
                    f"{edu.start_year}→{edu.end_year} ({edu.institution!r})"
                )
            if edu.duration_years > 12:
                r.add_warning(
                    f"education[{i}] duration > 12 years: "
                    f"{edu.duration_years}y ({edu.institution!r})"
                )
        if edu.end_year and edu.end_year > 2030:
            r.add_warning(f"education[{i}].end_year in far future: {edu.end_year}")

    # Overlapping education periods
    sorted_edu = sorted(
        [e for e in c.education if e.start_year and e.end_year],
        key=lambda e: e.start_year,  # type: ignore
    )
    for i in range(len(sorted_edu) - 1):
        e1, e2 = sorted_edu[i], sorted_edu[i + 1]
        if e2.start_year < e1.end_year:  # type: ignore
            r.add_warning(
                f"Overlapping education: {e1.institution!r} ({e1.start_year}–{e1.end_year}) "
                f"and {e2.institution!r} ({e2.start_year}–{e2.end_year})"
            )


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3: HONEYPOT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_honeypot(
    c: Candidate,
    r: ValidationResult,
    cfg: HoneypotConfig,
) -> None:
    """
    Detect suspicious patterns identified during dataset forensics.
    Each flag reduces the candidate's trust_score.

    Patterns detected:
      HP-01  SALARY_INVERTED
      HP-02  SKILL_DURATION_OVERFLOW
      HP-03  DATE_PARADOX
      HP-04  INFLATED_YOE
      HP-05  UNENDORSED_EXPERT
      HP-06  HIGH_SCORE_ZERO_ENDORSE
      HP-07  OVERLAP_JOBS
      HP-08  GHOST_PROFILE
      HP-09  TITLE_MISMATCH  (also in consistency, duplicated here with flag)
      HP-10  PCS_MISMATCH
    """
    rs = c.redrob_signals

    # HP-01: Salary inversion
    if rs.expected_salary.is_inverted:
        r.add_honeypot(
            "HP-01_SALARY_INVERTED",
            f"min={rs.expected_salary.min_lpa} > max={rs.expected_salary.max_lpa} LPA"
        )

    # HP-02: Skill duration > career duration
    total_career_months = c.total_career_months
    for skill in c.skills:
        if skill.duration_months > total_career_months + cfg.skill_duration_grace_months:
            r.add_honeypot(
                "HP-02_SKILL_DURATION_OVERFLOW",
                f"'{skill.name}' used {skill.duration_months}m "
                f"but career spans only {total_career_months}m"
            )

    # HP-03: Date paradox
    signup = _parse_date(rs.signup_date)
    active = _parse_date(rs.last_active_date)
    if signup and active and active < signup:
        r.add_honeypot(
            "HP-03_DATE_PARADOX",
            f"last_active={active} is before signup={signup}"
        )

    # HP-04: Inflated YoE
    claimed_yrs = c.profile.years_of_experience
    career_yrs  = c.implied_experience_years
    if claimed_yrs > career_yrs + cfg.yoe_inflation_threshold_years:
        r.add_honeypot(
            "HP-04_INFLATED_YOE",
            f"claimed={claimed_yrs:.1f}y vs career_sum={career_yrs:.1f}y "
            f"(gap={claimed_yrs - career_yrs:.1f}y)"
        )

    # HP-05: Expert skill with 0 endorsements (and 0 assessment)
    assessment_skills = set(rs.skill_assessment_scores.keys())
    for skill in c.skills:
        if skill.proficiency == "expert" and skill.endorsements < cfg.expert_min_endorsements:
            if skill.name not in assessment_skills:
                r.add_honeypot(
                    "HP-05_UNENDORSED_EXPERT",
                    f"'{skill.name}' is 'expert' with {skill.endorsements} endorsements "
                    f"and no platform assessment"
                )

    # HP-06: High assessment score, 0 endorsements for that skill
    skill_endorse = {s.name: s.endorsements for s in c.skills}
    for skill_name, score in rs.skill_assessment_scores.items():
        if score > cfg.assessment_zero_endorse_threshold:
            if skill_endorse.get(skill_name, 0) == 0:
                r.add_honeypot(
                    "HP-06_HIGH_SCORE_ZERO_ENDORSE",
                    f"'{skill_name}' assessment={score:.0f}/100 but 0 peer endorsements"
                )

    # HP-07: Overlapping full-time jobs (>3 months overlap)
    dated_jobs = sorted(
        [j for j in c.career_history if j.start_date and j.end_date],
        key=lambda j: j.start_date,  # type: ignore
    )
    for i in range(len(dated_jobs) - 1):
        j1_end  = dated_jobs[i].end_date
        j2_start = dated_jobs[i + 1].start_date
        if j1_end and j2_start and j2_start < j1_end:
            overlap_days = (
                datetime.strptime(j1_end, "%Y-%m-%d") -
                datetime.strptime(j2_start, "%Y-%m-%d")
            ).days
            overlap_months = overlap_days / 30.0
            if overlap_months > cfg.job_overlap_grace_months:
                r.add_honeypot(
                    "HP-07_OVERLAP_JOBS",
                    f"{dated_jobs[i].company!r} and {dated_jobs[i+1].company!r} "
                    f"overlap {overlap_months:.0f} months"
                )

    # HP-08: Ghost profile — invisible but highly active
    pv   = rs.profile_views_received_30d
    sa   = rs.search_appearance_30d
    sbr  = rs.saved_by_recruiters_30d
    apps = rs.applications_submitted_30d
    if pv == 0 and sa == 0 and sbr == 0 and apps > cfg.ghost_application_threshold:
        r.add_honeypot(
            "HP-08_GHOST_PROFILE",
            f"0 views / 0 searches / 0 saves but {apps} applications submitted"
        )

    # HP-10: Profile completeness score mismatch
    pcs   = rs.profile_completeness_score
    if pcs > 90 and c.n_skills < 3 and len(c.education) == 0:
        r.add_honeypot(
            "HP-10_PCS_MISMATCH",
            f"completeness={pcs:.0f}% but only {c.n_skills} skills and no education"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC BATCH VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def validate_batch(
    candidates: list,
    cfg: HoneypotConfig = HONEYPOT_CONFIG,
) -> list:
    """
    Validate a list of Candidate objects and return their ValidationResults.

    Args:
        candidates: List[Candidate]
        cfg:        HoneypotConfig (defaults to module-level HONEYPOT_CONFIG)

    Returns:
        List[ValidationResult] in the same order as the input.
    """
    return [validate_candidate(c, cfg) for c in candidates]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CLI SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.loader import load_sample

    print("Validator smoke test (100 candidates)...")
    sample = load_sample(n=100, validate=False)
    results = validate_batch(sample)

    flagged = [r for r in results if r.n_flags > 0]
    errors  = [r for r in results if not r.is_valid]
    print(f"  Validated : {len(results)}")
    print(f"  Errors    : {len(errors)}")
    print(f"  Flagged   : {len(flagged)}")
    if flagged:
        print(f"  Sample flag: {flagged[0]}")
    print("PASS")
