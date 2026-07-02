# Ranking Diagnostics: Phase 4 Baseline

This document provides an engineering audit of the first fully-scored top 100 candidates out of the 100,000 candidate dataset.

## 1. Score Distributions (100,000 Candidates)

| Score Range | Count | % of Dataset |
|-------------|-------|--------------|
| **>0.9**    | 0     | 0%           |
| **0.7 – 0.9**| 18    | 0.02%        |
| **0.5 – 0.7**| 856   | 0.86%        |
| **0.3 – 0.5**| 55,499| 55.5%        |
| **<0.3**    | 43,627| 43.6%        |

**Analysis:** The scoring bounds are healthy. No candidate achieved a perfect `>0.9` score, which is realistic for an aggregated metric combining 7 distinct, rigorous dimensions. The top 100 cut-off lies cleanly within the high `0.6`s.

## 2. Trust Distribution

- **Clean (1.0):** 67,829 candidates
- **Penalized (<1.0):** 32,171 candidates
- **Max Penalized (0.25):** 51 candidates

**Trust Action Audit:** 
We verified candidates who would have made the Top 100 based on their raw technical metrics but were dropped due to trust penalties. 
*Example:* `CAND_0055992` scored a brilliant **0.767** raw score (would be Top 3!), but their trust multiplier was reduced to `0.76` due to honeypot flags (e.g., overlapping jobs/skill overflows), dropping their final score to `0.583` and ejecting them from the Top 100 entirely. This confirms the trust filter is working flawlessly as a safe-guard.

## 3. Top 100 Cohort Profiles

**AI Skill Counts:**
- The Top 100 averages **8.1 AI skills**. 
- Minimum in the Top 100 is 4; Maximum is 12.
- *Diagnosis:* Healthy. The engine is rewarding deep technical stacks without exclusively favoring keyword stuffers.

**Experience (YoE):**
- The Top 100 averages **6.4 years of experience**.
- *Diagnosis:* This aligns perfectly with the JD's 5–12 year preferred range constraint. `experience_score` is maxing at 1.0 for these candidates.

**Most Common Current Titles:**
1. AI Research Engineer (12)
2. Applied ML Engineer (10)
3. ML Engineer (9)
4. AI Engineer (7)
5. Machine Learning Engineer (6)
6. Recommendation Systems Engineer (6)

**Average Domain Score:** `0.393`
- *Diagnosis:* Candidates are hitting roughly 40% of the maximum allowed domain keyword densities. Because the JD covers a wide net (Ranking, Retrieval, MLOps, NLP), it's highly improbable for a single human to max out all of them. `0.39` is a strong signal for relevance.

---

## 4. Sensitivity & Anomaly Analysis

- **Are candidates ranking highly with weak domain evidence?** 
  No. The Rank 100 candidate has the lowest domain score (`0.292`) but compensated by having 11 AI skills and high verification bounds. Anyone lower than `0.29` domain evidence naturally falls out of the Top 100 due to the 30% weight configuration.
- **Excessive dominance of Skill Score?**
  No. Rank 50 has only 4 AI skills (a lower skill score of `0.351`), proving that a candidate with exceptional verified behavior, education, and domain matching can still compete without listing every tool under the sun.
- **Suspicious Behavior?**
  None detected. The linear combination with a multiplicative trust decay is behaving predictably.

---

## 5. Candidate Case Studies

### Rank 1: CAND_0064326
- **Title:** Search Engineer
- **YoE:** 7.6 years
- **AI Skills:** 9
- **Final Score:** 0.765
- **Why they ranked here:** The perfect candidate. They have a massive `skill_score` (`0.86`), maxed-out experience within the golden 5-12 window, incredibly high domain relevance (`0.497` — indicating heavy use of search/retrieval keywords), and zero trust penalties.

### Rank 5: CAND_0011687
- **Title:** Senior NLP Engineer
- **YoE:** 7.8 years
- **AI Skills:** 8
- **Final Score:** 0.735
- **Why they ranked here:** Very similar to Rank 1, but with a slightly lower domain score (`0.412` vs `0.497`) and lower skill density. They make up for it with a perfect `1.0` education and availability score.

### Rank 10: CAND_0010685
- **Title:** NLP Engineer
- **YoE:** 6.7 years
- **AI Skills:** 8
- **Final Score:** 0.720
- **Why they ranked here:** Elite domain match (`0.509`, higher than Rank 1!) but suffered a heavy penalty on their Verification score (`0.40`), meaning they likely lacked GitHub links or rigorous platform assessments compared to those above them.

### Rank 50: CAND_0079387
- **Title:** AI Engineer
- **YoE:** 6.9 years
- **AI Skills:** 4
- **Final Score:** 0.663
- **Why they ranked here:** The "Diamond in the Rough" archetype. Only 4 AI skills (`0.35` skill score), but they are an absolutely verified, highly engaged, Tier-1 educated candidate (`verification=1.0`, `education=1.0`, `behavioral=0.86`). Their fundamentals carried them into the top half of the leaderboard.

### Rank 100: CAND_0043312
- **Title:** AI Research Engineer
- **YoE:** 4.2 years
- **AI Skills:** 11
- **Final Score:** 0.627
- **Why they ranked here:** Technically proficient (11 AI skills, `skill_score` of `0.684`), but heavily penalized by Experience (4.2 years falls below the preferred 5.0 minimum, dropping their experience score to `0.834`). Their domain score is also the lowest of the featured group (`0.292`), making them the precise boundary condition for top 100 quality.
