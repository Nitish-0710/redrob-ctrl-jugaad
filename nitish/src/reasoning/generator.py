"""
src/reasoning/generator.py
==========================
Deterministically generates recruiter-style reasoning for candidate ranking.
Output must be factual, concise, and non-hallucinatory, employing diverse
sentence structures based on candidate strengths.
"""

from typing import List
from src.data.parser import Candidate
from src.features.candidate_features import FeatureVector
from src.ranking.scorer import CandidateScore

def generate_reasoning(candidate: Candidate, fv: FeatureVector, score: CandidateScore) -> str:
    """
    Generate a 80-180 character deterministic reasoning string based on the
    candidate's actual metrics and ranking scores.
    """
    bd = score.score_breakdown
    exp = round(fv.implied_experience_years, 1)
    
    # ---------------------------------------------------------
    # 1. Determine Strongest Domain
    # ---------------------------------------------------------
    domain_map = {
        "search, retrieval, and ranking systems": max(fv.evidence_search, fv.evidence_retrieval, fv.evidence_ranking),
        "NLP and production ML": max(fv.evidence_nlp, fv.evidence_production_ml),
        "recommendation and personalization systems": max(fv.evidence_recommendation, fv.evidence_personalization),
        "applied ML": fv.evidence_machine_learning,
    }
    
    sorted_domains = sorted(domain_map.items(), key=lambda x: x[1], reverse=True)
    best_domain_label, best_domain_score = sorted_domains[0]
    if best_domain_score == 0:
        best_domain_label = "general software engineering"

    # ---------------------------------------------------------
    # 2. Determine Mild Concerns
    # ---------------------------------------------------------
    concern = ""
    if fv.trust_score < 1.0:
        concern = " Note: some profile attributes may benefit from additional verification."
    elif fv.notice_period_days >= 60:
        concern = f" Note: {fv.notice_period_days}-day notice period."
    elif fv.verification_score < 0.4:
        concern = " Note: limited verification signals."
    elif fv.ai_skill_count < 3:
        concern = " Note: limited core AI skills."

    # ---------------------------------------------------------
    # 3. Select Archetype Template (Waterfall based on strengths)
    # ---------------------------------------------------------
    skill_desc = f"{fv.ai_skill_count} core skills" if fv.ai_skill_count >= 5 else f"{fv.ai_skill_count} AI skills"
    
    reasoning = ""
    
    # Priority A: Verification-first (High assessment performance)
    if fv.verification_score >= 0.8 and fv.assessment_count > 0 and fv.avg_assessment_score >= 80:
        reasoning = f"High assessment performance and verified technical credentials complement {exp} years of {best_domain_label} experience."
        
    # Priority B: Domain-first (High domain score)
    elif bd.get("domain_score", 0) >= 0.4:
        if "NLP" in best_domain_label:
            reasoning = f"Strong {best_domain_label} background with {exp} years of experience. Demonstrates validated AI expertise through {skill_desc}."
        elif "recommendation" in best_domain_label:
            reasoning = f"Experience building {best_domain_label}, supported by strong AI skill coverage and technical verification."
        else:
            reasoning = f"Built {best_domain_label} across {exp} years of experience. Strong AI skill depth and verified technical activity."

    # Priority C: Skills-first (High AI skill count)
    elif fv.ai_skill_count >= 8:
        reasoning = f"Demonstrates strong technical depth with {skill_desc}. Backed by {exp} years of experience building {best_domain_label}."
        
    # Priority D: Behavioral-first (High engagement)
    elif bd.get("behavioral_score", 0) >= 0.8:
        reasoning = f"Highly engaged candidate with {exp} years of experience in {best_domain_label}. Shows solid proficiency across {skill_desc}."
        
    # Priority E: Experience-first (Fallback)
    else:
        reasoning = f"{exp} years of applied experience in {best_domain_label}. Solid technical profile including {skill_desc} and platform verification."

    # ---------------------------------------------------------
    # 4. Final Formatting
    # ---------------------------------------------------------
    final_reasoning = (reasoning + concern).strip()
    
    if len(final_reasoning) > 180:
        final_reasoning = final_reasoning[:177] + "..."
        
    return final_reasoning
