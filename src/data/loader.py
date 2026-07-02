"""
src/data/loader.py
==================
Responsible for loading candidates.jsonl from disk.

Supports three access patterns:
  1. load_all()     → List[Candidate]          — loads entire dataset into memory
  2. stream()       → Iterator[Candidate]       — one record at a time, O(1) memory
  3. load_chunks()  → Iterator[List[Candidate]] — configurable batch sizes

All loaders parse via parser.parse_candidate() and optionally validate
via validators.validate_candidate(). Malformed records are logged and skipped.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Generator, Iterable, Iterator, List, Optional, Tuple

from configs.paths import CANDIDATES_JSONL
from configs.settings import SETTINGS, get_logger
from src.data.parser import Candidate, parse_candidate

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

class LoadStats:
    """Tracks statistics of a load operation."""
    def __init__(self) -> None:
        self.total_lines:    int = 0
        self.parsed_ok:      int = 0
        self.parse_errors:   int = 0
        self.invalid:        int = 0
        self.elapsed_sec:    float = 0.0

    def __repr__(self) -> str:
        rate = self.parsed_ok / max(self.elapsed_sec, 1e-6)
        return (
            f"LoadStats(parsed={self.parsed_ok:,}, errors={self.parse_errors}, "
            f"invalid={self.invalid}, elapsed={self.elapsed_sec:.1f}s, "
            f"rate={rate:,.0f} rec/s)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def stream(
    path: Optional[Path] = None,
    max_records: Optional[int] = None,
    validate: bool = True,
    skip_invalid: bool = True,
) -> Generator[Candidate, None, None]:
    """
    Stream candidates one at a time from a JSONL file.

    Memory usage is O(1) — only one Candidate object is live at any time.
    Suitable for very large files (this dataset is 487 MB / 100K records).

    Args:
        path:         Path to the JSONL file. Defaults to CANDIDATES_JSONL.
        max_records:  Stop after this many successfully parsed records (None = all).
        validate:     Whether to run schema/honeypot validation after parsing.
        skip_invalid: If True, log and skip invalid records. If False, raise.

    Yields:
        Candidate objects in file order.
    """
    from src.data.validators import validate_candidate   # lazy import to avoid cycles

    filepath = path or CANDIDATES_JSONL
    count    = 0
    errors   = 0

    logger.info(f"Streaming from {filepath}")

    with open(filepath, "r", encoding=SETTINGS.file_encoding) as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            # ── Parse JSON ────────────────────────────────────────────────────
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                errors += 1
                logger.warning(f"Line {line_no}: JSON parse error — {exc}")
                if not skip_invalid:
                    raise
                continue

            # ── Convert to Candidate dataclass ────────────────────────────────
            try:
                candidate = parse_candidate(raw)
            except (KeyError, ValueError, TypeError) as exc:
                errors += 1
                cid = raw.get("candidate_id", f"<line {line_no}>")
                logger.warning(f"{cid}: parse_candidate failed — {exc}")
                if not skip_invalid:
                    raise
                continue

            # ── Optional validation ───────────────────────────────────────────
            if validate:
                result = validate_candidate(candidate)
                if not result.is_valid and not skip_invalid:
                    raise ValueError(
                        f"{candidate.candidate_id}: validation failed — {result.errors}"
                    )

            yield candidate
            count += 1

            if max_records is not None and count >= max_records:
                logger.info(f"Reached max_records={max_records:,}. Stopping.")
                break

    logger.info(f"Stream complete: {count:,} yielded, {errors} errors")


def load_all(
    path: Optional[Path] = None,
    max_records: Optional[int] = None,
    validate: bool = True,
    skip_invalid: bool = True,
    show_progress: bool = True,
) -> Tuple[List[Candidate], LoadStats]:
    """
    Load the entire dataset into memory as a list of Candidate objects.

    For 100K candidates this uses ~1–2 GB RAM depending on object overhead.
    Suitable for feature computation where random access is needed.

    Args:
        path:          Path to JSONL file. Defaults to CANDIDATES_JSONL.
        max_records:   Load at most this many records.
        validate:      Run validator on each record.
        skip_invalid:  Skip (True) or raise (False) on parse/validation errors.
        show_progress: Log progress every 10K records.

    Returns:
        Tuple of (List[Candidate], LoadStats)
    """
    stats    = LoadStats()
    results: List[Candidate] = []
    t0       = time.perf_counter()
    log_every = 10_000

    logger.info(f"Loading all candidates from {path or CANDIDATES_JSONL} ...")

    for i, candidate in enumerate(
        stream(path=path, max_records=max_records, validate=validate, skip_invalid=skip_invalid),
        start=1
    ):
        results.append(candidate)
        if show_progress and i % log_every == 0:
            elapsed = time.perf_counter() - t0
            logger.info(f"  Loaded {i:,} candidates ({elapsed:.1f}s elapsed)")

    stats.parsed_ok   = len(results)
    stats.elapsed_sec = time.perf_counter() - t0
    logger.info(f"Load complete: {stats}")
    return results, stats


def load_chunks(
    path: Optional[Path] = None,
    chunk_size: Optional[int] = None,
    max_records: Optional[int] = None,
    validate: bool = True,
    skip_invalid: bool = True,
) -> Generator[List[Candidate], None, None]:
    """
    Load candidates in configurable-size batches.

    Useful for pipeline stages that can process data in parallel chunks
    without requiring the full dataset in memory simultaneously.

    Args:
        path:        Path to JSONL file. Defaults to CANDIDATES_JSONL.
        chunk_size:  Records per batch. Defaults to SETTINGS.chunk_size.
        max_records: Total records cap across all chunks.
        validate:    Run validation per record.
        skip_invalid: Skip on error.

    Yields:
        Lists of Candidate objects, each of length ≤ chunk_size.
    """
    size   = chunk_size or SETTINGS.chunk_size
    chunk: List[Candidate] = []

    for candidate in stream(
        path=path, max_records=max_records,
        validate=validate, skip_invalid=skip_invalid
    ):
        chunk.append(candidate)
        if len(chunk) >= size:
            logger.debug(f"Yielding chunk of {len(chunk):,} candidates")
            yield chunk
            chunk = []

    if chunk:
        logger.debug(f"Yielding final chunk of {len(chunk):,} candidates")
        yield chunk


def load_sample(
    n: int = 1_000,
    path: Optional[Path] = None,
    validate: bool = True,
) -> List[Candidate]:
    """
    Quick helper: load a small sample of n candidates for dev/testing.

    Args:
        n:        How many candidates to load.
        path:     Optional path override.
        validate: Run validator.

    Returns:
        List of Candidate objects.
    """
    candidates, _ = load_all(
        path=path,
        max_records=n,
        validate=validate,
        show_progress=False,
    )
    logger.info(f"Sample loaded: {len(candidates)} candidates")
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT (quick sanity check)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    print("Running loader smoke test (first 100 candidates)...")
    candidates, stats = load_all(max_records=100, show_progress=False)
    print(f"  Loaded  : {len(candidates)}")
    print(f"  Stats   : {stats}")
    print(f"  Sample  : {candidates[0]}")
    print(f"  Skills  : {candidates[0].skill_names[:5]}")
    print("PASS")
