"""
src/ranking/jd_parser.py
========================
JD Understanding Layer

This module converts the fixed Senior AI Engineer Job Description into a
structured JobRequirements object that maps to candidate features.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# DATA SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RequiredDomains:
    retrieval: bool
    ranking: bool
    recommendation: bool
    search: bool
    relevance: bool
    personalization: bool
    evaluation: bool
    machine_learning: bool
    nlp: bool
    production_ml: bool


@dataclass
class ExperienceRequirements:
    minimum_years: float
    preferred_min_years: float
    preferred_max_years: float
    seniority_indicators: List[str]


@dataclass
class SkillRequirements:
    required_skills: List[str]
    preferred_skills: List[str]


@dataclass
class BehavioralPreferences:
    open_to_work: bool
    availability_days_max: int
    recruiter_engagement: bool


@dataclass
class EducationPreferences:
    degree_preferences: List[str]
    field_preferences: List[str]


@dataclass
class LocationPreferences:
    preferred_locations: List[str]
    relocation_preferred: bool


@dataclass
class JobRequirements:
    """
    Structured representation of the Job Description.
    This acts as the target query vector for ranking candidates.
    """
    role_title: str
    domains: RequiredDomains
    experience: ExperienceRequirements
    skills: SkillRequirements
    behavioral: BehavioralPreferences
    education: EducationPreferences
    location: LocationPreferences

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def parse_jd() -> JobRequirements:
    """
    Deterministically constructs the JobRequirements for the
    Senior AI Engineer role in the INDIA.RUNS 2026 Redrob AI Challenge.

    Since the JD is fixed, we avoid LLMs/APIs here to ensure 100%
    stability and reproducibility in the scoring phase.
    """
    return JobRequirements(
        role_title="Senior AI Engineer",
        
        domains=RequiredDomains(
            retrieval=True,
            ranking=True,
            recommendation=True,
            search=True,
            relevance=True,
            personalization=True,
            evaluation=True,
            machine_learning=True,
            nlp=True,
            production_ml=True,
        ),
        
        experience=ExperienceRequirements(
            minimum_years=3.0,
            preferred_min_years=5.0,
            preferred_max_years=12.0,
            seniority_indicators=["senior", "lead", "staff", "principal", "sr.", "sr"]
        ),
        
        skills=SkillRequirements(
            required_skills=[
                "Python", "Machine Learning", "Deep Learning", "NLP"
            ],
            preferred_skills=[
                "PyTorch", "Transformers", "LLMs", "Vector Databases",
                "MLOps", "Computer Vision", "Statistics"
            ]
        ),
        
        behavioral=BehavioralPreferences(
            open_to_work=True,
            availability_days_max=30,
            recruiter_engagement=True
        ),
        
        education=EducationPreferences(
            degree_preferences=["B.Tech", "M.Tech", "MS", "PhD"],
            field_preferences=[
                "Computer Science", "Artificial Intelligence",
                "Machine Learning", "Data Science", "Mathematics", "Statistics"
            ]
        ),
        
        location=LocationPreferences(
            preferred_locations=["India", "Remote"],
            relocation_preferred=True
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    # When run directly, generate the JSON artifact
    project_root = Path(__file__).resolve().parents[2]
    out_path = project_root / "job_requirements.json"
    
    reqs = parse_jd()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reqs.to_dict(), f, indent=4)
        
    print(f"Generated structured JD at: {out_path}")
