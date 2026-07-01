"""
src/scoring/honeypot_detector.py
=================================
Rule-based detector for impossible, contradictory, and synthetic candidate
profiles ("honeypots") in the Redrob dataset.

Architecture
------------
Every check is an independent pure function:

    _check_xxx(candidate, config) -> list[tuple[HoneypotFlag, float, str]]

Each tuple is (flag, confidence_0_to_1, human_readable_reason).
The `HoneypotDetector.detect()` method calls all checks, merges results,
and returns a single `HoneypotResult`.

This design ensures:
- Each check is unit-testable in isolation.
- Adding a new check requires zero changes to existing code.
- No ML models, no embeddings, no scoring logic — pure rule-based.

Confidence interpretation
-------------------------
  0.90 – 1.00 : Near-certain honeypot signal (e.g., expert skill + 0 months)
  0.70 – 0.89 : Strong signal (e.g., non-tech title + 5 advanced AI skills)
  0.50 – 0.69 : Moderate signal (e.g., summary/title mismatch keywords)
  0.30 – 0.49 : Weak / borderline signal (e.g., mild duration inflation)

is_honeypot decision rule
--------------------------
  True  if ANY single flag has confidence >= 0.85
  True  if the sum of all confidences >= 1.60
  False otherwise
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from typing import Callable

from config.settings import Config, DEFAULT_CONFIG
from src.models.candidate import CandidateRecord, SkillProficiency

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for a single check result item
# ---------------------------------------------------------------------------
CheckResult = tuple["HoneypotFlag", float, str]


# ---------------------------------------------------------------------------
# 1. HoneypotFlag — enumeration of all detectable anomaly types
# ---------------------------------------------------------------------------

class HoneypotFlag(Enum):
    """
    Enumeration of all honeypot / contradiction signal types.

    Each value is an auto()-assigned int for internal identity.
    The .name property (e.g., "SKILL_DURATION_INFLATION") is used in
    human-readable output.
    """

    # Skills list claims more cumulative months than the candidate's YOE allows
    SKILL_DURATION_INFLATION = auto()

    # A skill is claimed as "expert" or "advanced" but has negligible usage duration
    EXPERT_WITH_ZERO_EXPERIENCE = auto()

    # Non-technical current title paired with advanced AI/ML skills
    NON_TECH_WITH_ADVANCED_AI_SKILLS = auto()

    # The profile summary's language strongly contradicts the stated current title
    SUMMARY_TITLE_CONTRADICTION = auto()

    # A career history role's start/end dates are inconsistent with duration_months
    IMPOSSIBLE_TIMELINE = auto()

    # Career jumps from a completely unrelated domain to senior ML with no bridge roles
    SUSPICIOUS_CAREER_PROGRESSION = auto()

    # Current title is ML/AI but every career role is non-technical
    TITLE_CAREER_MISMATCH = auto()

    # Skills from mutually exclusive domains listed at advanced/expert level
    DOMAIN_SKILL_CONTRADICTION = auto()


# ---------------------------------------------------------------------------
# 2. HoneypotResult — output contract
# ---------------------------------------------------------------------------

@dataclass
class HoneypotResult:
    """
    Output of a single `HoneypotDetector.detect()` call.

    Fields
    ------
    is_honeypot : bool
        True if the candidate is classified as a honeypot / trap.
    confidence : float
        Aggregate confidence score (0.0 – 1.0). Computed as the maximum
        single-flag confidence, saturated at 1.0.
    flags : list[HoneypotFlag]
        All triggered flags (de-duplicated, in trigger order).
    reasons : list[str]
        Human-readable explanations corresponding to each flag.
    flag_confidences : dict[HoneypotFlag, float]
        Per-flag confidence for explainability and auditing.
    """

    is_honeypot:       bool
    confidence:        float
    flags:             list[HoneypotFlag]
    reasons:           list[str]
    flag_confidences:  dict[HoneypotFlag, float] = field(default_factory=dict)

    # Thresholds used for the decision (stored for audit trail)
    hard_threshold:    float = 0.85
    soft_sum_threshold: float = 1.60

    def summary(self) -> str:
        """One-line summary for logging."""
        if not self.flags:
            return "CLEAN (no flags)"
        flag_names = ", ".join(f.name for f in self.flags)
        return (
            f"{'HONEYPOT' if self.is_honeypot else 'SUSPICIOUS'} "
            f"(conf={self.confidence:.2f}) — [{flag_names}]"
        )


# ---------------------------------------------------------------------------
# 3. Keyword sets used across multiple checks
# ---------------------------------------------------------------------------

# AI/ML skills whose presence on a non-tech profile is suspicious
# Sourced from dataset_profile.md Tier B skills
_ADVANCED_AI_SKILL_KEYWORDS: frozenset[str] = frozenset({
    "embeddings", "vector search", "rag", "pinecone", "faiss",
    "semantic search", "sentence transformers", "hugging face transformers",
    "information retrieval", "recommendation systems", "llms",
    "fine-tuning llms", "langchain", "llama index", "llamaindex",
    "prompt engineering", "weaviate", "milvus", "qdrant", "pgvector",
    "opensearch", "elasticsearch", "bm25", "learning to rank", "ltr",
    "nlp", "pytorch", "tensorflow", "machine learning", "deep learning",
    "scikit-learn", "sklearn", "transformers", "bert", "gpt",
})

# CV / Speech keywords — wrong domain for this JD but not necessarily honeypots
_CV_SPEECH_KEYWORDS: frozenset[str] = frozenset({
    "computer vision", "image classification", "object detection",
    "yolo", "cnn", "opencv", "resnet", "speech recognition",
    "asr", "tts", "text to speech", "diffusion models", "gans",
})

# Summary phrases that betray a non-technical background
_NON_TECH_SUMMARY_PHRASES: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"my (professional )?background is in marketing",
        r"i('ve| have) spent my career in marketing",
        r"marketing manager",
        r"hr manager",
        r"content (writing|writer|strategy)",
        r"brand (design|identity|strategy)",
        r"sales (executive|manager|lead)",
        r"customer support",
        r"mechanical engineering design",
        r"accounting role",
        r"financial reporting",
        r"operations management",
        r"business analyst at a consulting",
        r"audit[- ]readiness",
        r"excel modeling",
    ]
)

# Title words that indicate non-technical roles
_NON_TECH_TITLE_KEYWORDS: frozenset[str] = frozenset({
    "marketing manager", "hr manager", "human resources",
    "accountant", "customer support", "sales executive",
    "content writer", "graphic designer", "operations manager",
    "civil engineer", "mechanical engineer", "project manager",
    "lawyer", "teacher", "finance manager", "brand manager",
})

# ML/AI title keywords — legitimate engineering roles
_ML_TITLE_KEYWORDS: frozenset[str] = frozenset({
    "machine learning", "ml engineer", "ai engineer", "nlp engineer",
    "data scientist", "applied ml", "search engineer",
    "recommendation", "ranking engineer", "retrieval",
    "research engineer", "staff ml", "principal ml", "ai specialist",
    "senior data scientist", "applied scientist",
})


# ---------------------------------------------------------------------------
# 4. Individual check functions (pure — no side effects)
# ---------------------------------------------------------------------------

def _check_skill_duration_inflation(
    candidate: CandidateRecord,
    config: Config,
) -> list[CheckResult]:
    """
    Flag candidates whose total claimed skill duration vastly exceeds
    what their years-of-experience could plausibly allow.

    Real engineers work on multiple skills simultaneously, so some
    inflation is normal (e.g., 2-3× is fine).  Beyond 6× is suspicious;
    beyond 10× is almost certainly synthetic padding.
    """
    results: list[CheckResult] = []

    total_skill_months = sum(sk.duration_months for sk in candidate.skills)
    yoe_months = candidate.total_yoe * 12.0

    if yoe_months <= 0:
        return results   # cannot evaluate without YOE

    ratio = total_skill_months / yoe_months

    borderline = config.honeypot.borderline_ratio        # e.g., 4.0
    hard_limit = config.honeypot.skill_duration_yoe_ratio  # e.g., 6.0

    if ratio >= hard_limit:
        # Scale confidence: 6× → 0.60, 10× → 1.0
        raw_conf = min((ratio - hard_limit) / (10.0 - hard_limit) + 0.60, 0.95)
        confidence = round(raw_conf, 3)
        reason = (
            f"Total claimed skill duration ({total_skill_months} months) is "
            f"{ratio:.1f}× the candidate's YOE ({yoe_months:.0f} months). "
            f"Threshold is {hard_limit}×.  Consistent with synthetic skill padding."
        )
        results.append((HoneypotFlag.SKILL_DURATION_INFLATION, confidence, reason))

    elif ratio >= borderline:
        confidence = round(0.30 + (ratio - borderline) / (hard_limit - borderline) * 0.25, 3)
        reason = (
            f"Skill duration ratio {ratio:.1f}× is elevated (borderline threshold "
            f"is {borderline}×).  May be normal parallelism or mild inflation."
        )
        results.append((HoneypotFlag.SKILL_DURATION_INFLATION, confidence, reason))

    return results


def _check_expert_with_zero_experience(
    candidate: CandidateRecord,
    config: Config,   # noqa: ARG001 — kept for uniform signature
) -> list[CheckResult]:
    """
    Flag 'expert' or 'advanced' skills with near-zero stated duration.

    A person cannot genuinely be 'expert' in a skill they have used for
    0–1 months.  This is a strong honeypot signal.
    """
    results: list[CheckResult] = []

    for sk in candidate.skills:
        if sk.proficiency == SkillProficiency.EXPERT and sk.duration_months == 0:
            reason = (
                f"Skill '{sk.name}' is claimed as EXPERT with 0 months of "
                f"usage.  This is logically impossible."
            )
            results.append((
                HoneypotFlag.EXPERT_WITH_ZERO_EXPERIENCE, 0.95, reason,
            ))

        elif sk.proficiency == SkillProficiency.EXPERT and sk.duration_months <= 2:
            reason = (
                f"Skill '{sk.name}' is claimed as EXPERT with only "
                f"{sk.duration_months} month(s) of usage."
            )
            results.append((
                HoneypotFlag.EXPERT_WITH_ZERO_EXPERIENCE, 0.85, reason,
            ))

        elif sk.proficiency == SkillProficiency.ADVANCED and sk.duration_months == 0:
            reason = (
                f"Skill '{sk.name}' is claimed as ADVANCED with 0 months of "
                f"usage — inconsistent."
            )
            results.append((
                HoneypotFlag.EXPERT_WITH_ZERO_EXPERIENCE, 0.80, reason,
            ))

        elif sk.proficiency == SkillProficiency.ADVANCED and sk.duration_months == 1:
            reason = (
                f"Skill '{sk.name}' is claimed as ADVANCED with only 1 month "
                f"of usage."
            )
            results.append((
                HoneypotFlag.EXPERT_WITH_ZERO_EXPERIENCE, 0.70, reason,
            ))

    return results


def _check_non_tech_title_with_ai_skills(
    candidate: CandidateRecord,
    config: Config,   # noqa: ARG001
) -> list[CheckResult]:
    """
    Flag non-technical current titles that claim advanced/expert AI skills.

    The JD explicitly warns: a Marketing Manager with Pinecone + RAG + Embeddings
    is the primary keyword-stuffing trap in the dataset.

    Confidence scales with:
    - Number of advanced/expert AI skills listed
    - How obviously non-technical the current title is
    """
    results: list[CheckResult] = []

    title_lower = candidate.current_title_lower

    # Determine if title is clearly non-technical
    is_non_tech = any(kw in title_lower for kw in _NON_TECH_TITLE_KEYWORDS)
    if not is_non_tech:
        return results

    # Count advanced/expert AI skills
    advanced_ai_skills: list[str] = []
    for sk in candidate.skills:
        sk_lower = sk.name.lower()
        if (
            any(kw in sk_lower for kw in _ADVANCED_AI_SKILL_KEYWORDS)
            and sk.proficiency in (SkillProficiency.ADVANCED, SkillProficiency.EXPERT)
        ):
            advanced_ai_skills.append(sk.name)

    if not advanced_ai_skills:
        return results

    n = len(advanced_ai_skills)
    # Confidence ramp: 1 skill → 0.55, 3 skills → 0.75, 5+ skills → 0.88
    if n >= 5:
        confidence = 0.88
    elif n >= 3:
        confidence = 0.75
    elif n >= 2:
        confidence = 0.65
    else:
        confidence = 0.55

    skill_list = ", ".join(f"'{s}'" for s in advanced_ai_skills[:6])
    reason = (
        f"Current title '{candidate.profile.current_title}' is non-technical, "
        f"but the candidate claims {n} advanced/expert AI skill(s): {skill_list}.  "
        f"This pattern is explicitly called out as a keyword-stuffing trap in the JD."
    )
    results.append((HoneypotFlag.NON_TECH_WITH_ADVANCED_AI_SKILLS, confidence, reason))
    return results


def _check_summary_title_contradiction(
    candidate: CandidateRecord,
    config: Config,   # noqa: ARG001
) -> list[CheckResult]:
    """
    Flag profiles where the free-text summary betrays a different career
    identity than the stated current title.

    Heuristic: scan the summary for non-tech role phrases.  If found, and
    the current title is technical/ML, flag the contradiction.

    This detects synthetic profiles where the generator copied the wrong
    summary template into a technical-title candidate.
    """
    results: list[CheckResult] = []

    summary_lower = candidate.profile.summary.lower()
    title_lower   = candidate.current_title_lower

    title_is_technical = any(kw in title_lower for kw in _ML_TITLE_KEYWORDS) or any(
        t in title_lower for t in ["engineer", "developer", "scientist", "analyst"]
    )
    if not title_is_technical:
        # Non-technical title with a matching non-tech summary is not contradictory
        return results

    matched_phrases: list[str] = []
    for pattern in _NON_TECH_SUMMARY_PHRASES:
        m = pattern.search(summary_lower)
        if m:
            matched_phrases.append(m.group(0))

    if not matched_phrases:
        return results

    n = len(matched_phrases)
    confidence = min(0.50 + n * 0.08, 0.78)

    phrase_examples = "; ".join(f'"{p}"' for p in matched_phrases[:3])
    reason = (
        f"Profile summary contains {n} non-technical phrase(s) inconsistent "
        f"with the stated title '{candidate.profile.current_title}'.  "
        f"Phrases detected: {phrase_examples}.  "
        f"Suggests a synthetic generation error (wrong summary template)."
    )
    results.append((HoneypotFlag.SUMMARY_TITLE_CONTRADICTION, confidence, reason))
    return results


def _check_impossible_timeline(
    candidate: CandidateRecord,
    config: Config,
) -> list[CheckResult]:
    """
    Flag career roles where the stated `duration_months` is inconsistent
    with the difference between `start_date` and `end_date`.

    A discrepancy of more than `config.honeypot.timeline_mismatch_months`
    (default: 24 months) is flagged as suspicious.

    We allow generous tolerance because:
    - end_date may be approximate (e.g., month-level precision)
    - The dataset has some benign rounding errors
    """
    results: list[CheckResult] = []
    threshold = config.honeypot.timeline_mismatch_months

    REFERENCE_DATE = date(2026, 6, 1)

    for role in candidate.career_history:
        end = role.end_date if role.end_date is not None else REFERENCE_DATE
        actual_months = (
            (end.year - role.start_date.year) * 12
            + (end.month - role.start_date.month)
        )

        if actual_months < 0:
            # end_date is before start_date — fundamentally impossible
            reason = (
                f"Role '{role.title}' at '{role.company}': end_date "
                f"({end}) is before start_date ({role.start_date}).  "
                f"Impossible timeline."
            )
            results.append((HoneypotFlag.IMPOSSIBLE_TIMELINE, 0.92, reason))
            continue

        delta = abs(actual_months - role.duration_months)
        if delta > threshold:
            # Scale confidence: 24mo delta → 0.65, 48mo+ → 0.90
            confidence = min(0.65 + (delta - threshold) / 24.0 * 0.25, 0.90)
            reason = (
                f"Role '{role.title}' at '{role.company}': stated "
                f"duration_months={role.duration_months} but "
                f"start_date={role.start_date} to end_date={end} "
                f"implies {actual_months} months — delta of {delta} months "
                f"(threshold: {threshold} months)."
            )
            results.append((HoneypotFlag.IMPOSSIBLE_TIMELINE, round(confidence, 3), reason))

    return results


def _check_suspicious_career_progression(
    candidate: CandidateRecord,
    config: Config,   # noqa: ARG001
) -> list[CheckResult]:
    """
    Flag improbable career jumps — e.g., Customer Support → Senior AI Engineer
    with no intermediate technical roles.

    We check:
    1. Current title is senior ML/AI (or equivalent).
    2. Immediately prior roles are all clearly non-technical.
    3. No intermediate ML/data/engineering roles exist in career history.
    """
    results: list[CheckResult] = []

    title_lower = candidate.current_title_lower
    current_is_senior_ml = any(kw in title_lower for kw in _ML_TITLE_KEYWORDS)
    if not current_is_senior_ml:
        return results

    if not candidate.career_history:
        return results

    # Classify every career role
    def _is_non_tech_role(title: str) -> bool:
        t = title.lower()
        return any(kw in t for kw in _NON_TECH_TITLE_KEYWORDS)

    def _is_tech_role(title: str) -> bool:
        t = title.lower()
        return any(kw in t for kw in {
            "engineer", "developer", "scientist", "analyst",
            "architect", "devops", "sre", "data", "ml", "ai",
        })

    non_tech_roles   = [r for r in candidate.career_history if _is_non_tech_role(r.title)]
    tech_roles       = [r for r in candidate.career_history if _is_tech_role(r.title)]
    total_roles      = len(candidate.career_history)
    non_tech_count   = len(non_tech_roles)

    # Flag if ALL prior roles are non-technical with no bridge
    if non_tech_count == total_roles:
        reason = (
            f"Current title '{candidate.profile.current_title}' suggests senior ML, "
            f"but ALL {total_roles} career role(s) are non-technical "
            f"({', '.join(r.title for r in non_tech_roles[:3])}).  "
            f"No technical bridge roles found."
        )
        results.append((HoneypotFlag.SUSPICIOUS_CAREER_PROGRESSION, 0.75, reason))

    elif non_tech_count > 0 and len(tech_roles) == 0:
        # Catch case where roles don't match either classifier
        reason = (
            f"Current title '{candidate.profile.current_title}' is an ML role, "
            f"but {non_tech_count}/{total_roles} career roles appear non-technical "
            f"with no identifiable technical progression."
        )
        results.append((HoneypotFlag.SUSPICIOUS_CAREER_PROGRESSION, 0.60, reason))

    return results


def _check_title_career_mismatch(
    candidate: CandidateRecord,
    config: Config,   # noqa: ARG001
) -> list[CheckResult]:
    """
    Flag candidates whose current title is ML/AI but whose entire career
    history is composed of non-technical industries/roles.

    Distinguishes from SUSPICIOUS_CAREER_PROGRESSION by focusing on
    industry/company context rather than role title sequence.
    """
    results: list[CheckResult] = []

    title_lower = candidate.current_title_lower
    title_is_ml = any(kw in title_lower for kw in _ML_TITLE_KEYWORDS)
    if not title_is_ml:
        return results

    if not candidate.career_history:
        return results

    services_industries = {"it services", "consulting", "bpo", "staffing",
                           "manufacturing", "paper products", "conglomerate"}
    totally_non_ml_industries = {"manufacturing", "paper products",
                                 "retail", "fmcg", "hospitality", "real estate"}

    non_ml_role_count = sum(
        1 for role in candidate.career_history
        if role.industry.lower() in totally_non_ml_industries
    )

    total = len(candidate.career_history)
    if non_ml_role_count == total and total >= 2:
        confidence = min(0.55 + non_ml_role_count * 0.08, 0.80)
        industries = [r.industry for r in candidate.career_history[:3]]
        reason = (
            f"Title '{candidate.profile.current_title}' is ML/AI, but all "
            f"{total} career role(s) are in non-ML industries "
            f"({', '.join(industries)}).  "
            f"No ML/software/tech industry context in career history."
        )
        results.append((HoneypotFlag.TITLE_CAREER_MISMATCH, round(confidence, 3), reason))

    return results


def _check_domain_skill_contradiction(
    candidate: CandidateRecord,
    config: Config,   # noqa: ARG001
) -> list[CheckResult]:
    """
    Flag profiles that claim advanced/expert proficiency in mutually
    exclusive technical domains simultaneously.

    Example: advanced Computer Vision (YOLO, CNN, OpenCV) AND advanced
    NLP/Retrieval (Pinecone, FAISS, Embeddings).  Real engineers specialise.

    Note: This is a softer signal.  Some senior engineers genuinely span
    domains.  We only flag when both counts are high.
    """
    results: list[CheckResult] = []

    ai_retrieval_count = 0
    cv_speech_count = 0

    for sk in candidate.skills:
        if sk.proficiency not in (SkillProficiency.ADVANCED, SkillProficiency.EXPERT):
            continue
        sk_lower = sk.name.lower()
        if any(kw in sk_lower for kw in _ADVANCED_AI_SKILL_KEYWORDS):
            ai_retrieval_count += 1
        if any(kw in sk_lower for kw in _CV_SPEECH_KEYWORDS):
            cv_speech_count += 1

    if ai_retrieval_count >= 4 and cv_speech_count >= 4:
        confidence = min(0.45 + (ai_retrieval_count + cv_speech_count) * 0.02, 0.70)
        reason = (
            f"Claims {ai_retrieval_count} advanced NLP/Retrieval skills AND "
            f"{cv_speech_count} advanced CV/Speech skills simultaneously.  "
            f"Real specialists rarely have deep expertise across both domains.  "
            f"Suggests synthetic skill assignment from multiple persona templates."
        )
        results.append((HoneypotFlag.DOMAIN_SKILL_CONTRADICTION, round(confidence, 3), reason))

    return results


# ---------------------------------------------------------------------------
# 5. Registry — ordered list of all check functions
# ---------------------------------------------------------------------------

# Type alias for a check function
CheckFn = Callable[[CandidateRecord, Config], list[CheckResult]]

_ALL_CHECKS: list[CheckFn] = [
    _check_skill_duration_inflation,
    _check_expert_with_zero_experience,
    _check_non_tech_title_with_ai_skills,
    _check_summary_title_contradiction,
    _check_impossible_timeline,
    _check_suspicious_career_progression,
    _check_title_career_mismatch,
    _check_domain_skill_contradiction,
]


# ---------------------------------------------------------------------------
# 6. HoneypotDetector — the public interface
# ---------------------------------------------------------------------------

class HoneypotDetector:
    """
    Stateless detector that runs all honeypot checks against a CandidateRecord.

    Usage
    -----
    ```python
    detector = HoneypotDetector()                  # uses DEFAULT_CONFIG
    result   = detector.detect(candidate_record)

    if result.is_honeypot:
        print(f"Honeypot detected: {result.summary()}")
    ```

    The detector is stateless — it holds only config references and is
    safe to reuse across the entire 100K candidate stream.
    """

    # Decision thresholds
    HARD_CONFIDENCE_THRESHOLD: float = 0.85   # single flag → honeypot
    SOFT_SUM_THRESHOLD:        float = 1.60   # sum of all flags → honeypot

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self._config = config
        logger.debug(
            "HoneypotDetector initialised with %d checks, "
            "hard_threshold=%.2f, soft_sum=%.2f",
            len(_ALL_CHECKS),
            self.HARD_CONFIDENCE_THRESHOLD,
            self.SOFT_SUM_THRESHOLD,
        )

    def detect(self, candidate: CandidateRecord) -> HoneypotResult:
        """
        Run all checks against `candidate` and return an aggregated result.

        Parameters
        ----------
        candidate : CandidateRecord
            A fully-parsed candidate (output of `parse_record`).

        Returns
        -------
        HoneypotResult
            Aggregated result with flags, confidences, and explanations.
        """
        all_results: list[CheckResult] = []

        for check_fn in _ALL_CHECKS:
            try:
                check_results = check_fn(candidate, self._config)
                all_results.extend(check_results)
            except Exception as exc:   # noqa: BLE001 — never crash the pipeline
                logger.warning(
                    "Check %s raised an exception for %s: %s",
                    check_fn.__name__,
                    candidate.candidate_id,
                    exc,
                    exc_info=True,
                )

        # ── Aggregate ─────────────────────────────────────────────────────────
        if not all_results:
            return HoneypotResult(
                is_honeypot=False,
                confidence=0.0,
                flags=[],
                reasons=[],
                flag_confidences={},
                hard_threshold=self.HARD_CONFIDENCE_THRESHOLD,
                soft_sum_threshold=self.SOFT_SUM_THRESHOLD,
            )

        # Collect flags, taking MAX confidence per flag (de-duplicate)
        flag_max_conf: dict[HoneypotFlag, float] = {}
        flag_reasons:  dict[HoneypotFlag, list[str]] = {}

        for flag, conf, reason in all_results:
            if flag not in flag_max_conf or conf > flag_max_conf[flag]:
                flag_max_conf[flag] = conf
            flag_reasons.setdefault(flag, []).append(reason)

        flags      = list(flag_max_conf.keys())
        confidences = list(flag_max_conf.values())
        conf_sum   = sum(confidences)
        max_conf   = max(confidences)
        aggregate_confidence = min(max_conf, 1.0)

        # Flatten reasons preserving order
        flat_reasons: list[str] = []
        for flag in flags:
            flat_reasons.extend(flag_reasons[flag])

        # ── Decision ──────────────────────────────────────────────────────────
        is_honeypot = (
            max_conf >= self.HARD_CONFIDENCE_THRESHOLD
            or conf_sum >= self.SOFT_SUM_THRESHOLD
        )

        result = HoneypotResult(
            is_honeypot=is_honeypot,
            confidence=round(aggregate_confidence, 4),
            flags=flags,
            reasons=flat_reasons,
            flag_confidences=flag_max_conf,
            hard_threshold=self.HARD_CONFIDENCE_THRESHOLD,
            soft_sum_threshold=self.SOFT_SUM_THRESHOLD,
        )

        if is_honeypot:
            logger.debug(
                "HONEYPOT %s: %s", candidate.candidate_id, result.summary()
            )

        return result

    def detect_batch(
        self,
        candidates: list[CandidateRecord],
        *,
        log_every: int = 10_000,
    ) -> dict[str, HoneypotResult]:
        """
        Run detection across a list of candidates.

        Returns a dict of candidate_id → HoneypotResult.
        Designed for use after `load_all_candidates()`.
        """
        results: dict[str, HoneypotResult] = {}
        n_honeypots = 0

        for i, candidate in enumerate(candidates, start=1):
            result = self.detect(candidate)
            results[candidate.candidate_id] = result
            if result.is_honeypot:
                n_honeypots += 1
            if i % log_every == 0:
                logger.info(
                    "Honeypot scan: %d/%d candidates processed, "
                    "%d honeypots found so far",
                    i, len(candidates), n_honeypots,
                )

        logger.info(
            "Honeypot scan complete: %d honeypots / %d total candidates (%.1f%%)",
            n_honeypots,
            len(candidates),
            100 * n_honeypots / max(len(candidates), 1),
        )
        return results
