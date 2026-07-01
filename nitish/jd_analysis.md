# JD Analysis: Senior AI Engineer

This document serves as the bridge between the human-readable Job Description and the structured `FeatureVector` extracted in Phase 2. It maps the implicit and explicit requirements of the JD to the exact candidate features that will be used in the ranking engine.

## 1. Core Role Definition

**Role:** Senior AI Engineer
**Primary Goal:** Build and optimize retrieval, ranking, and recommendation systems for intelligent candidate discovery.

---

## 2. JD Breakdown

### What the JD Explicitly Requires (Hard Constraints)
- **Minimum Experience:** At least 3–5 years of professional experience in data science, ML, or software engineering.
- **Core Skills:** Proficiency in Python, Machine Learning, Deep Learning, and NLP.
- **Domain Expertise:** Demonstrated experience building or interacting with Search, Retrieval, Ranking, and Recommendation engines.
- **Production Capability:** Experience taking ML models to production and evaluating them offline/online.

### What the JD Prefers (Soft Boosts)
- **Preferred Skills:** PyTorch, Transformers, LLMs, Vector Databases (FAISS, Milvus, etc.), MLOps.
- **Education:** B.Tech/M.Tech/MS/PhD in Computer Science, Artificial Intelligence, Machine Learning, Data Science, or related quantitative fields.
- **Seniority:** Evidence of technical leadership (titles including Senior, Lead, Staff, or Principal).
- **Behavioral & Availability:** 
  - Actively open to work (`open_to_work_flag`).
  - Short notice period (immediate joiners preferred).
  - High responsiveness to recruiters.
- **Location/Mobility:** Based in India/Remote, or willing to relocate.

### What the JD Discourages (Penalties & Disqualifiers)
- **Irrelevance:** Candidates with 0 AI skills (detected as 52.6% of the dataset in Phase 1).
- **Honeypot/Fraud Signals:** Inflated YoE, inverted salary expectations, overlapping full-time jobs, and unendorsed "expert" claims.
- **Poor Engagement:** Ghost profiles or extremely low interview completion rates.

---

## 3. JD to Feature Mapping

This is the exact mapping between the structured `JobRequirements` and the `FeatureVector` fields. This mapping informs the design of the Phase 4 Ranking Engine.

### 3.1 Domain Expertise Mapping

| Job Requirement Domain | FeatureVector Source |
|------------------------|----------------------|
| Retrieval | `evidence_retrieval` |
| Ranking | `evidence_ranking` |
| Recommendation | `evidence_recommendation` |
| Search | `evidence_search` |
| Relevance | `evidence_relevance` |
| Personalization | `evidence_personalization` |
| Evaluation / NDCG | `evidence_evaluation` |
| Machine Learning | `evidence_machine_learning` |
| NLP | `evidence_nlp` |
| Production ML / MLOps | `evidence_production_ml` |

### 3.2 Experience & Seniority Mapping

| JD Requirement | FeatureVector Source |
|----------------|----------------------|
| Min 5 Years YoE | `implied_experience_years` (preferred over claimed YoE for safety) |
| Stable job history | `avg_job_duration_months`, `max_job_duration_months` |
| Seniority Indicators | (Implicitly boosts `years_experience` and `trust_score`) |

### 3.3 Skill & Proficiency Mapping

| JD Requirement | FeatureVector Source |
|----------------|----------------------|
| Required Core Skills | `ai_skill_count`, `total_skill_count` |
| Advanced Proficiency | `proficiency_weighted_skill_score`, `expert_skill_count`, `advanced_skill_count` |
| Peer Validated Skills | `endorsement_weighted_skill_score` |

### 3.4 Education Mapping

| JD Requirement | FeatureVector Source |
|----------------|----------------------|
| CS/AI/Quantitative Degree | `cs_degree_flag`, `ai_degree_flag` |
| Top-tier Institutions | `highest_education_tier`, `tier1_count` |

### 3.5 Behavioral & Availability Mapping

| JD Requirement | FeatureVector Source |
|----------------|----------------------|
| Open to work | `open_to_work_flag` |
| Can join quickly | `notice_period_days`, `availability_score` |
| Engages with recruiters | `recruiter_response_rate`, `interview_completion_rate`, `engagement_score` |
| Willing to relocate | `willing_to_relocate` |

### 3.6 Verification & Trust Mapping

| JD Requirement | FeatureVector Source |
|----------------|----------------------|
| Validated technical chops | `github_activity_score`, `avg_assessment_score`, `verification_score` |
| Honest representation | `trust_score` (multiplier), `honeypot_flag_count` |
| Accurate timelines | Inverse of `experience_gap` and `inflated_experience_flag` |

---

## Conclusion
The structured `JobRequirements` JSON and the `FeatureVector` layer are now perfectly aligned. In Phase 4, the ranking engine will use these mappings to compute a relevance score via dot products, weighted sums, or learning-to-rank models.
