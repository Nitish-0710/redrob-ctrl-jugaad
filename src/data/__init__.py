"""
src/data/__init__.py
"""
from .parser import (
    Candidate, CandidateProfile, JobEntry, EducationEntry,
    Skill, Certification, Language, SalaryRange, RedrobSignals,
    parse_candidate,
)
from .loader import stream, load_all, load_chunks, load_sample, LoadStats
from .validators import validate_candidate, validate_batch, ValidationResult

__all__ = [
    "Candidate", "CandidateProfile", "JobEntry", "EducationEntry",
    "Skill", "Certification", "Language", "SalaryRange", "RedrobSignals",
    "parse_candidate",
    "stream", "load_all", "load_chunks", "load_sample", "LoadStats",
    "validate_candidate", "validate_batch", "ValidationResult",
]
