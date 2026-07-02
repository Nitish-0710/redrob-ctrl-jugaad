"""
src/utils/misc.py — Math/numeric helpers
"""
from typing import Optional


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def normalize(value: float, lo: float, hi: float) -> float:
    """
    Min-max normalize value to [0, 1].
    Returns 0.0 if lo == hi (degenerate case).
    """
    if hi <= lo:
        return 0.0
    return clamp((value - lo) / (hi - lo), 0.0, 1.0)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning default on division by zero."""
    if denominator == 0:
        return default
    return numerator / denominator
