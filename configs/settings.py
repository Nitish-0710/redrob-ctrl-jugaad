"""
configs/settings.py
====================
Runtime settings and environment configuration.

Separates operational concerns (logging level, chunk size, parallelism)
from domain concerns (feature weights, thresholds → feature_config.py)
and path concerns (all paths → paths.py).
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str) -> logging.Logger:
    """Return a pre-configured logger for a module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineSettings:
    """All runtime settings for the data loading and processing pipeline."""

    # ── Loading ───────────────────────────────────────────────────────────────
    # Number of candidates to load per chunk (streaming mode)
    chunk_size: int = 5_000

    # Maximum candidates to load (None = all)
    max_candidates: Optional[int] = None

    # Encoding for JSONL files
    file_encoding: str = "utf-8"

    # ── Validation ────────────────────────────────────────────────────────────
    # Whether to run validation on load (can be disabled for speed)
    validate_on_load: bool = True

    # Whether to skip malformed records (True) or raise exceptions (False)
    skip_invalid_records: bool = True

    # ── Caching ───────────────────────────────────────────────────────────────
    # Cache parsed candidates to parquet for faster subsequent runs
    use_cache: bool = True

    # Overwrite existing cache even if fresh
    force_reprocess: bool = False

    # ── Parallelism ───────────────────────────────────────────────────────────
    # Number of parallel workers for feature computation
    # 1 = sequential (safer for debugging); -1 = use all CPU cores
    n_workers: int = 1

    # ── Reference date ────────────────────────────────────────────────────────
    # Competition reference date (June 23, 2026)
    reference_date: str = "2026-06-23"

    # ── Submission ────────────────────────────────────────────────────────────
    # Number of candidates to include in submission ranking
    submission_top_k: int = 100

    # ── Debug / Dev ───────────────────────────────────────────────────────────
    # If True, only load a small sample for fast iteration
    dev_mode: bool = False
    dev_sample_size: int = 1_000


# ── Global singleton (override with environment or direct mutation) ────────────
SETTINGS = PipelineSettings(
    dev_mode=os.environ.get("DEV_MODE", "false").lower() == "true",
    max_candidates=int(os.environ["MAX_CANDIDATES"]) if "MAX_CANDIDATES" in os.environ else None,
    chunk_size=int(os.environ.get("CHUNK_SIZE", "5000")),
    n_workers=int(os.environ.get("N_WORKERS", "1")),
)
