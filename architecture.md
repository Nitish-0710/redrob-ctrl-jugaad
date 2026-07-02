# Architecture — INDIA.RUNS 2026 RedrRob AI Challenge

> **System**: Intelligent Candidate Discovery & Ranking  
> **Evaluation**: NDCG@10  
> **Status**: Phase 1 (Foundation) complete  

---

## 1. System Overview

The system takes `candidates.jsonl` (100K candidates, 487 MB) as input and produces `submission.csv` — a ranked list of the top-100 candidates for a **Senior AI Engineer** role, with scores and reasoning.

```
candidates.jsonl
      │
      ▼
 ┌────────────┐
 │   Loader   │  stream() / load_all() / load_chunks()
 └─────┬──────┘
       │  raw JSON dict
       ▼
 ┌────────────┐
 │   Parser   │  JSON → Candidate dataclass
 └─────┬──────┘
       │  Candidate
       ▼
 ┌────────────┐
 │ Validator  │  Schema + Consistency + Honeypot Detection
 └─────┬──────┘
       │  Candidate + ValidationResult (trust_score)
       ▼
 ┌────────────┐
 │  Features  │  Candidate → FeatureVector (Phase 2)
 └─────┬──────┘
       │  FeatureVector
       ▼
 ┌────────────┐
 │   Ranker   │  FeatureVector → score (Phase 4)
 └─────┬──────┘
       │  scored candidates
       ▼
 ┌────────────┐
 │  Reasoning │  score → human-readable string (Phase 5)
 └─────┬──────┘
       │
       ▼
  submission.csv
```

---

## 2. Directory Structure

```
project/
├── configs/                    # All configuration (single source of truth)
│   ├── __init__.py
│   ├── paths.py                # Filesystem paths
│   ├── feature_config.py       # Feature weights, thresholds, skill taxonomies
│   └── settings.py             # Runtime settings (chunk size, logging, etc.)
│
├── data/
│   ├── raw/                    # Symlink or copy of candidates.jsonl (never modify)
│   ├── interim/                # Intermediate artifacts (e.g., filtered subsets)
│   └── processed/              # Cached parquet files (features.parquet, etc.)
│
├── notebooks/
│   └── dataset_analysis.ipynb  # Phase 1: Forensic analysis (100K records)
│
├── outputs/
│   └── submission.csv          # Final ranked submission
│
├── src/
│   ├── __init__.py
│   ├── data/                   # [IMPLEMENTED] Data pipeline foundation
│   │   ├── __init__.py
│   │   ├── loader.py           # JSONL loading (stream/bulk/chunk)
│   │   ├── parser.py           # JSON → Candidate dataclass
│   │   └── validators.py       # Schema + honeypot detection
│   │
│   ├── features/               # [STUB] Phase 2: Feature engineering
│   │   ├── __init__.py
│   │   └── candidate_features.py   # FeatureVector schema + extract_features()
│   │
│   ├── retrieval/              # [STUB] Phase 3: Pre-filtering
│   ├── ranking/                # [STUB] Phase 4: Scoring & ranking
│   ├── reasoning/              # [STUB] Phase 5: Reasoning generation
│   ├── evaluation/             # [STUB] Phase 6: NDCG evaluation
│   └── utils/
│       ├── __init__.py
│       ├── io.py               # Parquet read/write
│       └── misc.py             # clamp, normalize, safe_divide
│
├── tests/
│   ├── __init__.py
│   ├── test_parser.py          # Unit tests for parser
│   └── test_validators.py      # Unit tests for validators
│
├── requirements.txt
├── README.md
└── architecture.md             # This file
```

---

## 3. Module Responsibilities

### 3.1 `configs/paths.py`
Single source of truth for all filesystem paths. **No other module hardcodes paths.**  
Import pattern: `from configs.paths import CANDIDATES_JSONL, PROCESSED_DIR`

### 3.2 `configs/feature_config.py`
Central registry for every domain constant:
- `AI_CORE_SKILLS` — the 38-skill taxonomy for the target role
- `PROFICIENCY_WEIGHTS` — expert=4, advanced=3, intermediate=2, beginner=1
- `HONEYPOT_CONFIG` — immutable dataclass with all detection thresholds
- `EDUCATION_TIER_WEIGHTS`, `BEHAVIORAL_WEIGHTS`, `NORMALIZATION_BOUNDS`

**Rule**: Feature engineers modify this file; downstream code just reads it.

### 3.3 `configs/settings.py`
Runtime configuration: chunk size, logging, dev mode, reference date.  
Supports environment-variable overrides (`DEV_MODE=true`, `CHUNK_SIZE=1000`).

### 3.4 `src/data/parser.py`
Converts raw JSON dicts to typed Python dataclasses:
```
Candidate
├── CandidateProfile
├── List[JobEntry]
├── List[EducationEntry]
├── List[Skill]
├── List[Certification]
├── List[Language]
└── RedrobSignals
    └── SalaryRange
```
All fields have safe defaults — parser never raises on missing keys.  
Derived properties: `implied_experience_years`, `experience_discrepancy`, `highest_edu_tier`.

### 3.5 `src/data/loader.py`
Three access patterns:
| Function | Memory | Use Case |
|----------|--------|----------|
| `stream()` | O(1) | Large dataset iteration |
| `load_all()` | O(n) | Feature matrix computation |
| `load_chunks()` | O(chunk) | Parallel batch processing |
| `load_sample()` | O(n) | Dev/testing |

### 3.6 `src/data/validators.py`
Three-layer validation per candidate:
1. **Schema** — field presence, type correctness, enum membership
2. **Consistency** — date ordering, salary range, title match, education timelines
3. **Honeypot** — 10 patterns from forensic analysis:

| Code | Name | Found in Dataset |
|------|------|-----------------|
| HP-01 | SALARY_INVERTED | 18,865 (18.9%) |
| HP-02 | SKILL_DURATION_OVERFLOW | 16,500 (16.5%) |
| HP-03 | DATE_PARADOX | 7,496 (7.5%) |
| HP-04 | INFLATED_YOE | 25 |
| HP-05 | UNENDORSED_EXPERT | 24 |
| HP-06 | HIGH_SCORE_ZERO_ENDORSE | 7 |
| HP-07 | OVERLAP_JOBS | varies |
| HP-08 | GHOST_PROFILE | varies |
| HP-09 | TITLE_MISMATCH | in warnings |
| HP-10 | PCS_MISMATCH | rare |

Each flag reduces `trust_score` by 12%, floored at 0.25.

---

## 4. Data Flow — Detailed

```
Step 1: LOAD
  loader.stream(CANDIDATES_JSONL)
  └── reads line by line (O(1) memory)

Step 2: PARSE
  parse_candidate(raw_json_dict)
  └── returns Candidate dataclass

Step 3: VALIDATE
  validate_candidate(candidate)
  └── returns ValidationResult
      ├── errors:         List[str]   (blocking)
      ├── warnings:       List[str]   (non-blocking)
      ├── honeypot_flags: List[str]
      └── trust_score:    float [0.25, 1.0]

Step 4: FEATURIZE  [Phase 2]
  extract_features(candidate, validation)
  └── returns FeatureVector
      ├── ai_skill_score
      ├── experience_score
      ├── education_score
      ├── assessment_score
      ├── github_score
      ├── trust_multiplier   ← from ValidationResult
      ├── behavioral_score
      └── availability_score

Step 5: RANK  [Phase 4]
  Scorer.score(feature_vector)
  └── returns float in [0, 1]

Step 6: GENERATE REASONING  [Phase 5]
  ReasoningGenerator.generate(candidate, feature_vector, score)
  └── returns str (≤200 chars for submission)

Step 7: BUILD SUBMISSION
  SubmissionBuilder.build(ranked_candidates[:100])
  └── writes submission.csv
```

---

## 5. Design Decisions

### Why dataclasses over dicts?
- **Type safety** — IDE autocomplete, mypy checking
- **Derived properties** — `implied_experience_years` computed once, reused everywhere
- **Explicit schema** — every field documented at the class level
- **Testability** — easy to construct fixture objects in tests

### Why three loader modes?
- **stream()** — the dataset is 487 MB; random-access loading isn't always needed
- **load_all()** — feature computation benefits from random access across all candidates
- **load_chunks()** — future parallelism (e.g., multiprocessing pool)

### Why validators are separate from parser?
- Parser: "Can I read this?" — never raises on missing data
- Validator: "Is this trustworthy?" — domain logic, honeypot detection
- Separation allows validation to be skipped for speed (e.g., when loading cached data)

### Why feature_config.py has a frozen HoneypotConfig?
- Immutable configuration prevents accidental mutation during pipeline execution
- Easy to instantiate alternative configs for ablation studies

---

## 6. Implementation Phases

| Phase | Module | Status | Description |
|-------|--------|--------|-------------|
| 1 | `data/` | ✅ **DONE** | Load, parse, validate |
| 2 | `features/` | 🔜 Next | Feature extraction & scoring functions |
| 3 | `retrieval/` | Planned | Hard filtering before ranking |
| 4 | `ranking/` | Planned | Weighted scoring + sort |
| 5 | `reasoning/` | Planned | Human-readable ranking justification |
| 6 | `evaluation/` | Planned | NDCG@k computation & ablation |

---

## 7. Key Forensic Findings (Phase 1)

From `notebooks/dataset_analysis.ipynb`:

| Finding | Value | Impact |
|---------|-------|--------|
| Total candidates | 100,000 | Full dataset |
| Candidates with 0 AI skills | 52,580 (52.6%) | Pre-ranked to bottom |
| Candidates with 5+ AI skills | 6,381 (6.4%) | Real competitive pool |
| Salary inversions | 18,865 (18.9%) | Do NOT use raw salary |
| Skill duration overflows | 16,500 (16.5%) | Discount duration features |
| Date paradoxes | 7,496 (7.5%) | Trust penalty |
| Avg assessment score coverage | 24.2% | High-value selective signal |
| GitHub linked | 35.4% | Strong differentiator |
| Missing values | 1 field (avg_assessment_score: 75.76%) | Expected; handle as None |

---

## 8. Running the Pipeline

```bash
# Smoke test: parse + validate 100 candidates
python src/data/loader.py

# Run unit tests
python tests/test_parser.py
python tests/test_validators.py

# Dev mode (1,000 candidates only)
DEV_MODE=true python src/data/loader.py

# Validate paths
python configs/paths.py
```
