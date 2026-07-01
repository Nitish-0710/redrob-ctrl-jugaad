# Feature Catalog — Phase 2

This catalog documents the flat `FeatureVector` representation extracted from parsed candidates. These features form the foundation for all downstream ranking and retrieval tasks.

## 1. Experience Features

| Feature Name | Description | Source | Type | Range | Missing Value Handling |
|--------------|-------------|--------|------|-------|------------------------|
| `years_experience` | Explicit YoE claimed on profile | `profile.years_of_experience` | float | 0.0 – 50.0 | Defaults to 0.0 |
| `implied_experience_years` | Total career duration / 12 | `career_history` | float | 0.0 – 50.0 | Defaults to 0.0 |
| `experience_gap` | Claimed YoE minus implied YoE | Derived | float | -50.0 – 50.0 | N/A |
| `total_jobs` | Number of distinct jobs held | `career_history` | int | 0 – ~20 | Defaults to 0 |
| `avg_job_duration_months` | Average duration per job | Derived | float | 0.0 – ~600.0 | Defaults to 0.0 |
| `max_job_duration_months` | Longest job held | Derived | int | 0 – ~600 | Defaults to 0 |
| `min_job_duration_months` | Shortest job held | Derived | int | 0 – ~600 | Defaults to 0 |

## 2. Skill Features

| Feature Name | Description | Source | Type | Range | Missing Value Handling |
|--------------|-------------|--------|------|-------|------------------------|
| `total_skill_count` | Total number of skills listed | `skills` | int | 0 – ~50 | Defaults to 0 |
| `ai_skill_count` | Number of skills in AI core taxonomy | `skills` + config | int | 0 – 38 | Defaults to 0 |
| `expert_skill_count` | Number of expert-level skills | `skills` | int | 0 – ~50 | Defaults to 0 |
| `advanced_skill_count` | Number of advanced-level skills | `skills` | int | 0 – ~50 | Defaults to 0 |
| `intermediate_skill_count` | Number of intermediate-level skills | `skills` | int | 0 – ~50 | Defaults to 0 |
| `beginner_skill_count` | Number of beginner-level skills | `skills` | int | 0 – ~50 | Defaults to 0 |
| `proficiency_weighted_skill_score` | Sum of proficiency weights for AI skills | Derived | float | 0.0 – ~150.0 | Defaults to 0.0 |
| `endorsement_weighted_skill_score` | Weight * min(endorsements, 50) for AI skills | Derived | float | 0.0 – ~7500.0 | Defaults to 0.0 |

## 3. Career Evidence Features
*These features use `log(1 + count)` normalization on text from headline, summary, and job descriptions.*

| Feature Name | Description | Source | Type | Range | Missing Value Handling |
|--------------|-------------|--------|------|-------|------------------------|
| `evidence_retrieval` | Frequency of retrieval/search keywords | Derived | float | 0.0+ | Defaults to 0.0 |
| `evidence_ranking` | Frequency of ranking/NDCG keywords | Derived | float | 0.0+ | Defaults to 0.0 |
| `evidence_recommendation` | Frequency of recommendation keywords | Derived | float | 0.0+ | Defaults to 0.0 |
| `evidence_search` | Frequency of search engine keywords | Derived | float | 0.0+ | Defaults to 0.0 |
| `evidence_relevance` | Frequency of relevance keywords | Derived | float | 0.0+ | Defaults to 0.0 |
| `evidence_personalization` | Frequency of personalization keywords | Derived | float | 0.0+ | Defaults to 0.0 |
| `evidence_evaluation` | Frequency of offline evaluation keywords | Derived | float | 0.0+ | Defaults to 0.0 |
| `evidence_machine_learning` | Frequency of general ML/DL keywords | Derived | float | 0.0+ | Defaults to 0.0 |
| `evidence_nlp` | Frequency of NLP/LLM keywords | Derived | float | 0.0+ | Defaults to 0.0 |
| `evidence_production_ml` | Frequency of MLOps/Scale keywords | Derived | float | 0.0+ | Defaults to 0.0 |

## 4. Education Features

| Feature Name | Description | Source | Type | Range | Missing Value Handling |
|--------------|-------------|--------|------|-------|------------------------|
| `education_count` | Number of degrees listed | `education` | int | 0 – ~10 | Defaults to 0 |
| `highest_education_tier` | Highest prestige tier of any degree | `education` | string | tier_1-4, unknown | Defaults to 'unknown' |
| `tier1_count` | Number of Tier 1 degrees | `education` | int | 0 – ~5 | Defaults to 0 |
| `tier2_count` | Number of Tier 2 degrees | `education` | int | 0 – ~5 | Defaults to 0 |
| `tier3_count` | Number of Tier 3 degrees | `education` | int | 0 – ~5 | Defaults to 0 |
| `tier4_count` | Number of Tier 4 degrees | `education` | int | 0 – ~5 | Defaults to 0 |
| `cs_degree_flag` | Has CS or related degree | Derived | int | 0 or 1 | Defaults to 0 |
| `ai_degree_flag` | Has explicit AI/ML/DS degree | Derived | int | 0 or 1 | Defaults to 0 |

## 5. Behavioral Features

| Feature Name | Description | Source | Type | Range | Missing Value Handling |
|--------------|-------------|--------|------|-------|------------------------|
| `recruiter_response_rate` | Rate of responding to recruiters | `redrob_signals` | float | 0.0 – 1.0 | Defaults to 0.0 |
| `interview_completion_rate` | Rate of completing scheduled interviews | `redrob_signals` | float | 0.0 – 1.0 | Defaults to 0.0 |
| `offer_acceptance_rate` | Rate of accepting extended offers | `redrob_signals` | float | -1.0 – 1.0 | Defaults to -1.0 |
| `profile_views_30d` | Profile views in last 30 days | `redrob_signals` | int | 0+ | Defaults to 0 |
| `applications_30d` | Applications submitted in last 30 days | `redrob_signals` | int | 0+ | Defaults to 0 |
| `search_appearance_30d` | Search appearances in last 30 days | `redrob_signals` | int | 0+ | Defaults to 0 |
| `saved_by_recruiters_30d` | Times saved by recruiters in last 30 days | `redrob_signals` | int | 0+ | Defaults to 0 |
| `endorsements_received` | Total platform endorsements | `redrob_signals` | int | 0+ | Defaults to 0 |
| `open_to_work_flag` | Open to new opportunities | `redrob_signals` | int | 0 or 1 | Defaults to 0 |
| `willing_to_relocate` | Willing to relocate for job | `redrob_signals` | int | 0 or 1 | Defaults to 0 |
| `engagement_score` | Composite score (Response + Interview + PCS) | Derived | float | 0.0 – 1.0 | Defaults to 0.0 |

## 6. Verification Features

| Feature Name | Description | Source | Type | Range | Missing Value Handling |
|--------------|-------------|--------|------|-------|------------------------|
| `verified_email` | Has verified email | `redrob_signals` | int | 0 or 1 | Defaults to 0 |
| `verified_phone` | Has verified phone | `redrob_signals` | int | 0 or 1 | Defaults to 0 |
| `linkedin_connected` | Linked LinkedIn account | `redrob_signals` | int | 0 or 1 | Defaults to 0 |
| `github_available` | Has valid GitHub score | `redrob_signals` | int | 0 or 1 | Defaults to 0 |
| `github_activity_score` | GitHub activity score | `redrob_signals` | float | 0.0 – 100.0 | Defaults to 0.0 |
| `assessment_count` | Number of assessments taken | `redrob_signals` | int | 0+ | Defaults to 0 |
| `avg_assessment_score` | Average score across assessments | `redrob_signals` | float | 0.0 – 100.0 | Defaults to 0.0 |
| `verification_score` | Composite verification strength | Derived | float | 0.0 – 1.0 | Defaults to 0.0 |

## 7. Availability Features

| Feature Name | Description | Source | Type | Range | Missing Value Handling |
|--------------|-------------|--------|------|-------|------------------------|
| `notice_period_days` | Notice period required | `redrob_signals` | int | 0 – 180 | Defaults to 60 |
| `availability_score` | Decaying score based on notice + open_to_work | Derived | float | 0.0 – 1.2 | Defaults to 0.0 |

## 8. Trust Features

| Feature Name | Description | Source | Type | Range | Missing Value Handling |
|--------------|-------------|--------|------|-------|------------------------|
| `trust_score` | Multiplicative penalty based on honeypot flags | `ValidationResult` | float | 0.25 – 1.0 | Defaults to 1.0 |
| `honeypot_flag_count` | Total number of honeypots tripped | `ValidationResult` | int | 0 – 10 | Defaults to 0 |
| `salary_inverted_flag` | Tripped HP-01 (Salary Inversion) | `ValidationResult` | int | 0 or 1 | Defaults to 0 |
| `skill_duration_overflow_flag` | Tripped HP-02 (Skill > Career Duration) | `ValidationResult` | int | 0 or 1 | Defaults to 0 |
| `date_paradox_flag` | Tripped HP-03 (Activity before signup) | `ValidationResult` | int | 0 or 1 | Defaults to 0 |
| `inflated_experience_flag` | Tripped HP-04 (YoE > Implied YoE + 3) | `ValidationResult` | int | 0 or 1 | Defaults to 0 |
| `unendorsed_expert_flag` | Tripped HP-05 (Expert with 0 endorsements) | `ValidationResult` | int | 0 or 1 | Defaults to 0 |
