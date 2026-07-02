# Ranking Engine Analysis: Phase 4

This document outlines the first iteration of the heuristic ranking engine designed to evaluate candidates for the Senior AI Engineer role. It clearly separates facts (derived from data), our assumptions, hypotheses for future tuning, and the scoring formulation.

## 1. Scoring Formula

The ranking engine uses a weighted linear combination of 7 normalized sub-scores, which is then scaled by a trust multiplier.

### Base Score Formula
```text
Base Score = (w1 * Domain Score) + 
             (w2 * Skill Score) + 
             (w3 * Experience Score) + 
             (w4 * Behavioral Score) + 
             (w5 * Verification Score) + 
             (w6 * Education Score) + 
             (w7 * Availability Score)
```

### Final Score
```text
Final Score = Base Score * Trust Multiplier
```
- `Trust Multiplier` is bounded between `[0.25, 1.0]`. A completely clean candidate receives `1.0` (no penalty), while each flagged honeypot issue reduces the multiplier by `12%` down to the hard floor of `0.25`.

---

## 2. Weight Choices & Rationale

The weights (defined in `configs/ranking_config.py`) reflect a typical recruiter's priority hierarchy for a specialized Senior AI role.

| Component | Weight | Rationale |
|-----------|--------|-----------|
| **Domain Score** | 30% | Hardest requirement. General ML engineers without specific retrieval/search/ranking backgrounds will struggle. Keyword matches in career history directly prove relevance. |
| **Skill Score** | 25% | Technical foundation. A candidate must possess the core AI tooling (Python, LLMs, PyTorch). |
| **Experience Score** | 15% | Ensures seniority. We specifically target the 5–12 year range. Overqualified candidates (12+ years) are mildly penalized as they may not be hands-on or might be too expensive. |
| **Behavioral Score** | 10% | A brilliant engineer who ignores recruiters or misses interviews is useless to the hiring pipeline. |
| **Verification Score** | 10% | Differentiates strong engineers who have GitHub portfolios or validated assessments from those who merely list keywords. |
| **Education Score** | 5% | A baseline filter. Nice to have top-tier CS degrees, but less important than actual domain experience for a senior role. |
| **Availability Score**| 5% | Immediate joiners are slightly preferred, but we won't heavily penalize a standard 60-day notice period for top talent. |v

---

## 3. Separation of Knowledge

### FACTS (Knowns from Dataset Forensics)
- **52.6% of candidates have 0 AI skills.** This makes the `skill_score` an incredibly effective initial filter.
- **18.9% of candidates have inverted salary ranges.** This is a confirmed data quality issue handled by the `trust_score`.
- **~75% of candidates have NO assessment scores.** We cannot heavily weight assessment scores without penalizing the vast majority of the pool. Therefore, `verification_score` relies on multiple fallbacks (GitHub, verified email/phone, LinkedIn).

### ASSUMPTIONS (Heuristics we designed)
- **Assumption 1:** `log(1 + x)` is an appropriate normalization for keyword counts in the `domain_score`. (We cap this evidence score at `3.0` which corresponds to roughly `e^3 - 1 ≈ 19` keyword occurrences).
- **Assumption 2:** Overlap in full-time jobs, inverted salaries, and extreme duration mismatches are indicative of fraudulent or low-quality profiles (Honeypots), warranting a hard penalty on the final score.
- **Assumption 3:** Peer endorsements matter, but self-claimed proficiency matters more. The `skill_score` splits this `70% / 30%`.

### HYPOTHESES (For Future Tuning)
- **Hypothesis A:** The current 30% weight on `domain_score` might be *too* high if the TF-IDF / keyword parsing captures too much noise (e.g., someone mentioning "search" in the context of "search engine optimization" rather than "vector search").
- **Hypothesis B:** We may need an absolute cutoff threshold (e.g., `Final Score > 0.40`) to generate the top 100, rather than just taking the top 100 sorted values, to ensure quality.
- **Hypothesis C:** The `experience_score` penalty for >12 years of experience might inadvertently penalize extremely experienced leaders who are willing to be hands-on.

---

## 4. Future Tuning Opportunities

1. **Ablation Studies:** Run the scorer across the dataset, systematically dropping one component weight to `0` and observing the shift in the top-100 cohort.
2. **TF-IDF over Keyword Counting:** Replace simple substring matching in the `domain_score` with a proper BM25/TF-IDF vectorizer if the current log-counts prove too noisy.
3. **Hard Filtering (Phase 3 Integration):** Before scoring 100,000 candidates, we should use a hard retrieval step (e.g., `ai_skill_count > 0` AND `trust_score > 0.5`) to reduce compute.
