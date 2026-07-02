"""
configs/paths.py
================
Single source of truth for all filesystem paths.

All modules must import paths from here — never use hardcoded strings.
"""

from pathlib import Path

# ── Project root (one level above configs/) ──────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Raw challenge data (candidates.jsonl lives here) ─────────────────────────
if (PROJECT_ROOT / "candidates.jsonl").exists():
    CHALLENGE_DATA_DIR: Path = PROJECT_ROOT
else:
    CHALLENGE_DATA_DIR: Path = PROJECT_ROOT.parent   # fallback to sibling/parent of project

CANDIDATES_JSONL:   Path = CHALLENGE_DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES:  Path = CHALLENGE_DATA_DIR / "sample_candidates.json"
SAMPLE_SUBMISSION:  Path = CHALLENGE_DATA_DIR / "sample_submission.csv"
CANDIDATE_SCHEMA:   Path = CHALLENGE_DATA_DIR / "candidate_schema.json"

# ── Internal data dirs ────────────────────────────────────────────────────────
DATA_DIR:           Path = PROJECT_ROOT / "data"
RAW_DIR:            Path = DATA_DIR / "raw"
INTERIM_DIR:        Path = DATA_DIR / "interim"
PROCESSED_DIR:      Path = DATA_DIR / "processed"

# ── Notebooks ─────────────────────────────────────────────────────────────────
NOTEBOOKS_DIR:      Path = PROJECT_ROOT / "notebooks"

# ── Outputs (submission CSVs, feature matrices, logs) ─────────────────────────
OUTPUTS_DIR:        Path = PROJECT_ROOT / "outputs"
SUBMISSION_CSV:     Path = OUTPUTS_DIR / "submission.csv"
FEATURE_CACHE:      Path = PROCESSED_DIR / "features.parquet"
CANDIDATE_CACHE:    Path = PROCESSED_DIR / "candidates.parquet"

# ── Configs ───────────────────────────────────────────────────────────────────
CONFIGS_DIR:        Path = PROJECT_ROOT / "configs"

# ── Src ───────────────────────────────────────────────────────────────────────
SRC_DIR:            Path = PROJECT_ROOT / "src"


def ensure_dirs() -> None:
    """Create all required directories if they don't exist yet."""
    for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Project paths:")
    print(f"  PROJECT_ROOT      : {PROJECT_ROOT}")
    print(f"  CANDIDATES_JSONL  : {CANDIDATES_JSONL}  (exists={CANDIDATES_JSONL.exists()})")
    print(f"  PROCESSED_DIR     : {PROCESSED_DIR}")
    print(f"  OUTPUTS_DIR       : {OUTPUTS_DIR}")
