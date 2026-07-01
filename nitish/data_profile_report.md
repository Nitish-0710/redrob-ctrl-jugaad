# INDIA.RUNS 2026 — RedrRob AI Challenge
# Data Profile Report

> **Generated**: 2026-06-23 16:12
> **Reference Date**: 2026-06-23
> **Analyst**: Dataset Forensics Pass v1.0
> **Dataset**: `candidates.jsonl` — 100,000 records, 487.3 MB

---

## 1. Key Findings

| Metric | Value |
|--------|-------|
| Total Candidates | 100,000 |
| Total Skill Entries | 960,302 |
| Total Education Records | 139,778 |
| Total Career Job Entries | 300,171 |
| JSON Parse Errors | 0 |
| Unique Skill Names | 133 |
| Unique Companies | 63 |
| Unique Countries | 8 |
| Unique Locations | 28 |

---

## 2. Data Quality Concerns

### 2.1 Missing Values

Only **one field** has missing values across the entire dataset:

| Field | Missing Count | Missing % |
| --- | --- | --- |
| `avg_assessment_score` | 75,756 | 75.76% |

> This is expected — `avg_assessment_score` is `NaN` when a candidate has taken **zero platform assessments**. The 75.76% absence means most candidates haven't done any Redrob skill assessments — this field is a highly selective signal.

### 2.2 ID Integrity

- **Duplicate candidate_ids**: 0 — ✅ Clean
- **Malformed IDs**: 0 — ✅ All follow `CAND_XXXXXXX` format
- **Sequential ID gaps**: 0 — ✅ IDs are perfectly sequential 1–100,000
- **Near-duplicate profiles**: 29,265 (29.3%) — same title + company + YoE + skill count

> The **29,265 near-duplicate profiles** are NOT actual duplicates — they're synthetic dataset artifacts where multiple candidates share identical surface attributes. This is a key challenge: **the signal must go deeper than surface fields**.

### 2.3 Structural Issues Detected

| Issue | Count | Severity |
|-------|-------|----------|
| Overlapping education periods | 13,808 | Medium |
| Salary inverted (max < min) | 18,865 | HIGH |
| Skill duration > career duration | 16,500 | HIGH |
| Date paradox (active < signup) | 7,496 | HIGH |
| Connections > 1000 | 273 | Low |
| Applications > 15/month | 219 | Low |
| Impossible edu timelines | 0 | None |
| Very short jobs (<3 months) | 0 | None |
| 0-skill candidates | 0 | None |

---

## 3. Candidate Quality Observations

### 3.1 Experience Distribution

- **Claimed YoE**: mean = 7.2 yrs (healthy spread across 0–50 years)
- **Implied YoE** (sum of job durations): mean = 7.1 yrs
- **Delta is tiny** (7.17 vs 7.06) — dataset is mostly self-consistent on experience
- **High discrepancy (|gap| > 5 yrs)**: only 39 candidates — very clean
- Most candidates cluster in the **2–14 year experience band**

### 3.2 AI/ML Skill Coverage (Critical for Ranking)

| Tier | Count | % of Pool | Notes |
|------|-------|-----------|-------|
| 0 AI core skills | 52,580 | 52.6% | Irrelevant profiles — rank at bottom |
| 1–4 AI skills | 41,039 | 41.0% | Adjacent/aspiring AI |
| 5–9 AI skills | 6,322 | 6.3% | Solid AI practitioners |
| 10+ AI skills | 59 | 0.1% | Elite AI specialists |

> **52.6% of the 100K candidate pool has ZERO AI core skills.** This is the most important filter — the entire bottom half of the ranking is essentially pre-determined for a Senior AI Engineer role.

**Top AI skills found**: `Feature Engineering` (6,240), `Statistical Modeling` (5,891), `NLP` (4,820), `Deep Learning` (4,150), `Machine Learning` (4,120), `Python` (3,910), `Image Classification` (3,450), `Speech Recognition` (3,230), `GANs` (2,980), `Computer Vision` (2,760)

### 3.3 Education Profile

  - **Tier-1**: ~3% of education records (IITs, IIMs, etc.)
  - **Tier-2**: ~12% of education records (NITs, good private universities)
  - **Tier-3**: ~35% of education records
  - **Tier-4**: ~50% of education records (Local Engineering Colleges)

- Most common field: **Computer Science** and **Information Technology**
- Most common degree: **B.Tech / B.E.** (dominant across the pool)
- Overlapping education periods: 13,808 — students doing concurrent programs (some valid, some suspicious)

### 3.4 Geographic Distribution

  - **India**: ~75,000 (75.0%)
  - **USA**: ~8,000 (8.0%)
  - **Canada**: ~5,000 (5.0%)
  - **UK**: ~4,000 (4.0%)
  - **Australia**: ~3,000 (3.0%)
  - **Singapore**: ~2,000 (2.0%)
  - **UAE**: ~2,000 (2.0%)
  - **Germany**: ~1,000 (1.0%)

> India dominates the candidate pool (~75%). Only 8 unique countries and 28 unique city locations — **very limited geographic diversity**. This suggests the dataset is India-centric with some diaspora profiles.

### 3.5 Engagement & Behavioral Signals

| Signal | Value | Interpretation |
|--------|-------|----------------|
| Open to work | 35.3% | Most NOT actively open — passive talent pool |
| Verified email | 72.0% | Decent verification rate |
| Verified phone | 61.8% | Moderate verification rate |
| LinkedIn linked | 36.0% | Low LinkedIn connectivity |
| Willing to relocate | 28.8% | Most prefer local roles |
| GitHub linked (score ≥ 0) | 35.4% | Only 35.4% have GitHub — huge differentiator |
| No GitHub (-1 sentinel) | 64.6% | ⚠️ Majority have no visible GitHub presence |
| Avg recruiter response rate | 0.437 | 43.7% response — moderate engagement |
| Avg interview completion | 0.620 | 62% — decent reliability |

### 3.6 Company Landscape

  - **TCS**: ~25,000+
  - **Wipro**: ~20,000+
  - **Infosys**: ~18,000+
  - **Accenture**: ~12,000+
  - **HCL Technologies**: ~8,000+
  - **Dunder Mifflin**: ~7,000+
  - **Acme Corp**: ~6,000+
  - **Stark Industries**: ~5,000+
  - **Globex Inc**: ~4,000+
  - **Initech**: ~3,000+

> Only **63 unique companies** across 300K+ job entries — extremely low diversity. The dataset heavily features Indian IT giants (TCS, Wipro, Infosys) and fictional companies (Dunder Mifflin, Acme Corp, Stark Industries, Globex Inc, Initech). The fictional companies are deliberate test fixtures in the synthetic dataset.

---

## 4. Potential Ranking Signals

### 4.1 Primary Discriminating Signals

| Signal | Type | Direction | Notes |
|--------|------|-----------|-------|
| `n_ai_skills` | Skill match | Higher = better | Most discriminative for AI role |
| `avg_assessment_score` | Verified skill | Higher = better | Only 24.2% have this — very high trust |
| `n_assessments` | Verification depth | More = better | Depth of platform verification |
| `github_activity_score` | Technical activity | Higher (ignore -1) | Strong for AI engineers |
| `current_title` | Role match | Fuzzy match to "AI/ML" | Seniority + domain signal |
| `years_of_experience` | Seniority | Optimal 5–12 yrs | Sweet spot for "Senior" role |

### 4.2 Trust & Credibility Signals

| Signal | Type | Direction | Notes |
|--------|------|-----------|-------|
| `endorsements_received` | Peer validation | Higher = better | Social proof |
| `verified_email + verified_phone` | Trust | Both true = high trust | Basic identity verification |
| `linkedin_connected` | Professional network | True = better | Platform credibility |
| `exp_discrepancy` | Honesty check | Near 0 = honest | Catch inflated YoE claims |
| `profile_completeness` | Profile quality | Higher = better | Effort signal |

### 4.3 Behavioral Engagement Signals

| Signal | Type | Direction | Notes |
|--------|------|-----------|-------|
| `recruiter_response_rate` | Responsiveness | Higher = better | Candidate seriousness |
| `interview_completion_rate` | Reliability | Higher = better | Follows through on commitments |
| `offer_acceptance_rate` | Selectivity | Moderate ideal | -1 = no history (neutral) |
| `saved_by_recruiters_30d` | External validation | Higher = better | Market demand signal |
| `open_to_work_flag` | Availability | True preferred | Immediate availability |
| `days_since_active` | Recency | Lower = better | Active candidates more responsive |

### 4.4 Correlation Insights (from heatmap)

- `years_of_experience` ↔ `implied_exp_years`: **r ≈ 0.98** — validates data integrity
- `profile_completeness` ↔ `n_skills`: **r ≈ 0.55** — completeness driven by skill count
- `profile_views_30d` ↔ `search_appearance_30d`: **r ≈ 0.70** — organic visibility loop
- `n_ai_skills` ↔ `recruiter_response_rate`: **low correlation** — skill count ≠ engagement quality
- `github_activity_score` ↔ `avg_assessment_score`: **low correlation** — independent verification dimensions

---

## 5. Honeypot Signal Analysis

### 5.1 Detection Summary

| Metric | Value |
|--------|-------|
| Candidates with ≥1 flag | 32,171 (32.2%) |
| Total flags raised | 42,917 |
| Avg flags per flagged candidate | 1.33 |

### 5.2 Flag Type Breakdown

| Flag Type | Count | % of Dataset | Description |
|-----------|-------|--------------|-------------|
| `SALARY_INVERTED` | 18,865 | 18.9% | Salary max < min — data error or fabrication |
| `SKILL_DURATION_OVERFLOW` | 16,500 | 16.5% | Skill usage months > total career months |
| `DATE_PARADOX` | 7,496 | 7.5% | Last active date BEFORE signup date |
| `INFLATED_YOE` | 25 | 0.03% | Claimed YoE >> career history sum |
| `UNENDORSED_EXPERT` | 24 | 0.024% | Expert skill, 0 endorsements |
| `HIGH_SCORE_ZERO_ENDORSE` | 7 | 0.0070% | Assessment >85 but no peer endorsements |

### 5.3 Critical Honeypot Findings

#### 🚨 SALARY_INVERTED — 18,865 candidates (18.9%)
The most prevalent data quality issue. Nearly **1 in 5 candidates** has a salary range where `max < min`. This is either:
- A systematic data generation error in the synthetic dataset
- A deliberate honeypot to test whether ranking systems use raw salary fields blindly
- **Action**: Always use `salary_min` only, or validate `max >= min` before using salary signals

#### 🚨 SKILL_DURATION_OVERFLOW — 16,500 candidates (16.5%)
Skills listed with usage duration **exceeding the candidate's entire career history**. For example, a 3-year career history with a skill listed as "48 months practiced." This is physically impossible.
- **Action**: Trust duration-weighted skill signals only when `skill_duration <= total_career_months`

#### 🚨 DATE_PARADOX — 7,496 candidates (7.5%)
Last active date is **earlier than signup date**. A clear data integrity error.
- **Action**: Flag for trust penalty; use `signup_date` as the floor for `last_active_date`

#### ⚠️ INFLATED_YOE — 25 candidates
Small set with claimed experience > 3 years beyond career history. True honeypot candidates.
- **Action**: Apply multiplicative penalty; use `min(claimed_yoe, implied_yoe + 1)` as the credible YoE

#### ⚠️ UNENDORSED_EXPERT — 24 candidates
Self-declared expert with zero peer endorsements. Suspicious credibility signal.
- **Action**: Discount expert proficiency if endorsements = 0 AND no assessment score

### 5.4 Most Suspicious Profile Examples

  **CAND_XXXXXXX** (6+ flags)
  - `SALARY_INVERTED:min>max`
  - `SKILL_DURATION_OVERFLOW:SomSkill>careerDuration`
  - `DATE_PARADOX:active<signup`
  - `INFLATED_YOE:claimed>>career`

  *(See honeypot_analysis.png for full ranked list)*


---

## 6. Recommendations for Feature Engineering

### Priority 1 — Core AI Relevance (Decisive Signal)

```python
# Weighted AI skill score
def ai_skill_score(candidate):
    PROF_WEIGHTS = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    score = 0
    for skill in candidate["skills"]:
        if skill["name"] in AI_CORE_SKILLS:
            prof_w = PROF_WEIGHTS.get(skill["proficiency"], 1)
            endorse_boost = min(1.0, skill["endorsements"] / 10)  # cap at 1.0
            score += prof_w * (0.7 + 0.3 * endorse_boost)
    return score
```

### Priority 2 — Trust Multipliers (Anti-Honeypot)

```python
def trust_multiplier(candidate):
    flags = 0
    rs = candidate["redrob_signals"]
    # Inverted salary penalty
    sal = rs.get("expected_salary_range_inr_lpa", {}) or {}
    if sal.get("max", 999) < sal.get("min", 0): flags += 1
    # YoE inflation
    claimed = candidate["profile"]["years_of_experience"]
    implied = sum(j["duration_months"] for j in candidate["career_history"]) / 12
    if claimed > implied + 3: flags += 2
    # Skill overflow
    total_months = sum(j["duration_months"] for j in candidate["career_history"])
    for s in candidate["skills"]:
        if s.get("duration_months", 0) > total_months + 12: flags += 0.5
    return max(0.3, 1.0 - 0.15 * flags)   # floor at 0.3
```

### Priority 3 — Behavioral Score

```python
def behavioral_score(rs):
    rrr = rs.get("recruiter_response_rate", 0)
    icr = rs.get("interview_completion_rate", 0)
    oar = max(0, rs.get("offer_acceptance_rate", 0))  # treat -1 as 0
    gh  = max(0, rs.get("github_activity_score", 0)) / 100  # normalize
    return 0.35*rrr + 0.30*icr + 0.15*oar + 0.20*gh
```

### Priority 4 — Engagement & Availability

```python
def availability_score(rs):
    otw      = 1.0 if rs.get("open_to_work_flag") else 0.5
    relocate = 1.1 if rs.get("willing_to_relocate") else 1.0
    notice   = max(0, 1.0 - rs.get("notice_period_days", 60) / 180)
    return otw * relocate * (0.5 + 0.5 * notice)
```

### Anti-patterns to Hard-Penalize
1. `n_ai_skills == 0` → Score floor (rank in bottom 60%)
2. `salary_inverted` → Ignore salary signals entirely for this candidate
3. `exp_discrepancy > 5 years` → Cap usable YoE at `implied_yoe`
4. `skill_duration > total_career_months` → Discount skill duration signals
5. `date_paradox` → Apply 15% trust penalty

---

## 7. Dataset Verdict & Readiness

| Aspect | Status | Verdict |
|--------|--------|---------|
| Data completeness | 1 field partially missing | ✅ READY |
| Schema compliance | 100% | ✅ CLEAN |
| Duplicate records | 0 exact duplicates | ✅ CLEAN |
| ID integrity | Perfect sequential IDs | ✅ CLEAN |
| Honeypot presence | 32.2% flagged candidates | ⚠️ SIGNIFICANT |
| AI candidate density | Only 6.4% have 5+ AI skills | ⚠️ SPARSE SIGNAL |
| Signal richness | 25+ signals available | ✅ RICH |
| Geographic diversity | 8 countries, India-dominant | ℹ️ NOTED |

### Final Recommendation

> **The dataset is synthetic and contains deliberate quality traps (18.9% salary inversions, 16.5% skill overflows, 7.5% date paradoxes).** These are likely honeypot tests embedded by the organizers to penalize naive rankers.
>
> The ranking strategy should:
> 1. **Hard-filter** by AI skill presence first (eliminate the 52.6% with 0 AI skills)
> 2. **Apply trust multipliers** to penalize honeypot-flagged candidates
> 3. **Use Redrob behavioral signals** (assessment score, response rate, GitHub) as tiebreakers
> 4. **Prioritize verified signals** over self-reported ones wherever possible
>
> The NDCG@10 will be dominated by getting the **top-10 ordering correct** — focus the feature engineering on the top-2% pool (candidates with 5+ AI skills + high assessment scores + active GitHub).
