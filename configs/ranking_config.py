"""
configs/ranking_config.py
=========================
Configuration for the candidate ranking engine.
Defines component weights, normalization bounds, and scoring thresholds.
"""

from typing import Dict

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
# These weights determine the final score formulation.
# Base Score = Sum(weight * normalized_component_score)
# Final Score = Base Score * Trust Multiplier

COMPONENT_WEIGHTS: Dict[str, float] = {
    "domain_score":       0.30,  # Highest weight: direct JD text match
    "skill_score":        0.25,  # Second highest: AI core skills
    "experience_score":   0.15,  # YoE matching ideal range
    "behavioral_score":   0.10,  # Engagement and response
    "verification_score": 0.10,  # Github, assessments, verified contacts
    "education_score":    0.05,  # Degree relevance and tier
    "availability_score": 0.05,  # Notice period & open-to-work
}

# Ensure base weights sum to 1.0 for easy reasoning
assert sum(COMPONENT_WEIGHTS.values()) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Domain evidence max values (log-scale counts from FeatureVector)
# Used to min-max normalize the domain score.
DOMAIN_MAX_EVIDENCE = 3.0

# Skill score max values (for normalization)
# Based on 99th percentile of dataset analysis
MAX_PROFICIENCY_SCORE = 30.0
MAX_ENDORSEMENT_SCORE = 1500.0


# ══════════════════════════════════════════════════════════════════════════════
# SCORING CONSTANTS & THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

# Penalty factor for YoE mismatch
EXPERIENCE_PENALTY_PER_YEAR = 0.1

# Minimum required score to be included in submission (soft threshold)
MINIMUM_RANKING_SCORE = 0.10
