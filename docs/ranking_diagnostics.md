# Ranking Diagnostics — Evolution, Issues, Corrections, and Lessons

> Documents the iterative development of the ranking system: what problems were found, how they were diagnosed, and what code changes resolved them.
> Every issue here is grounded in observed candidate behavior from the dataset.

---

## 1. Evolution Overview

```mermaid
timeline
    title Ranking Pipeline Evolution
    section Initial Version
        V1 Baseline : Keyword count scoring only
                    : Evidence score = sum of skill keywords
                    : No penalty gates
                    : No honeypot detection
    section Issue Discovery
        Audit Round 1 : Business Analysts flooding Top 20
                      : Non-technical titles appearing in Top 50
                      : Marketing Manager ranked #3 with "expert FAISS"
    section Fixes Applied
        V2 Scoring : Non-technical title gate added × 0.20
                   : Evidence floor gate added × 0.25
                   : Technical eligibility gate added × 0.05
    section Refinement
        V3 Calibration : BA exception threshold raised
                       : product_ml_experience_years two-tier model
                       : Honeypot detection formalised
    section Current
        V4 Trust Layer : trust_score multiplier added 0.70-1.00
                       : verification_score added
                       : Activity recency penalty added
                       : Reasoning format improved
```

---

## 2. Issue 1 — Business Analyst Flooding

### Problem Observed

In early versions, Business Analysts with career descriptions containing phrases like "machine learning pipeline", "the ML team built a recommendation system", and "oversees the AI/ML infrastructure" were scoring highly on `product_ml_experience_years`. This caused BAs to rank in the **Top 20**.

### Root Cause

The initial `product_ml_experience_years` calculation granted ML credit to any role at a product company whose description contained "machine learning", "PyTorch", or "neural network" — even when the role was a **Business Analyst describing the company's ML infrastructure** that other engineers had built.

```mermaid
graph LR
    BA["Business Analyst\nProduct Company\nDescription: 'worked with the ML team\nto define requirements for the\nrecommendation system built in PyTorch'"]

    BA -->|Old logic| OLD["General evidence: 'PyTorch' found\nCredit granted\nML YOE += role.duration_months"]
    BA -->|New logic| NEW{Is role non-technical?}
    NEW -->|Yes| STRONG{Strong evidence?\nFAISS / embedding / vector search}
    STRONG -->|No| NOCREDIT["No ML credit"]
    STRONG -->|Yes| CREDIT["ML credit granted\n(exceptional BA-to-ML transition)"]
```

### Fix Applied

Introduced a **two-tier evidence model** in `_extract_career_quality()`:

- **Strong evidence** keywords (FAISS, embedding, vector search, transformer, sentence_transformer): ML credit granted regardless of title — these are technical implementation words a BA wouldn't naturally use.
- **General evidence** keywords (machine learning, PyTorch, deep learning): ML credit only if `role_is_non_tech == False`.

### Before / After

| Scenario | Before | After |
|---|---|---|
| BA describing ML team's work | ML YOE credited, high Career Fit | No ML YOE; Career Fit stays at base 50 |
| BA who genuinely built FAISS index | ML YOE credited | ML YOE credited (strong evidence override) |
| Real ML Engineer at product company | ML YOE credited | ML YOE credited (unchanged) |

---

## 3. Issue 2 — Non-Technical Title Penalty Too Lenient

### Problem Observed

After implementing the initial non-technical title penalty (gate × 0.20), a relief exception was added for candidates who could prove real ML substance. The initial exception thresholds were:

```python
# Old exception (too lenient)
if product_ml_experience_years >= 3.0 AND evidence >= 60:
    # skip penalty
```

This allowed qualified-seeming BAs with padded career descriptions to reach the Top 30.

### Root Cause

`product_ml_experience_years >= 3.0` was achievable by a BA at a product company whose descriptions mentioned enough ML words. Combined with `evidence >= 60` being easily met with moderately padded text, the exception was too permissive.

### Fix Applied

Raised all three exception thresholds simultaneously:

```mermaid
graph TD
    OLD["Old Exception\nproduct_ml_yoe >= 3.0\nAND evidence >= 60"] -->|Too many BAs escaping| PROBLEM["BAs appearing in Top 30"]

    PROBLEM -->|Fix| NEW["New Exception\nproduct_ml_yoe >= 5.0\nAND evidence >= 75\nAND technical_fit >= 70"]

    NEW --> RESULT["Only genuine BA-to-ML transitions\nwith strong 3-axis evidence pass\nAll others take the 80% penalty"]
```

### Before / After

| Candidate Profile | Before | After |
|---|---|---|
| BA, 4yr product company, evidence=65, tech_fit=60 | Exception granted, no penalty | All three thresholds fail → 80% penalty |
| Real ML Engineer with non-standard title, 5yr prod ML | Exception granted | Still passes (higher scores across all three) |
| BA, 6yr product company, genuine ML builder, evidence=80 | Exception granted | Still passes (rare but legitimate) |

---

## 4. Issue 3 — Honeypot Candidates in Early Submissions

### Problem Observed

Before the `HoneypotDetector` was formalised, several profiles with impossible skill declarations were reaching the Top 100:

- A "Recommendation Systems Engineer" with 677 months of total skill usage on 6 YOE (84.6× ratio)
- Profiles claiming "expert" in FAISS with `duration_months=0`
- Marketing Managers with expert Pinecone, RAG, and Embeddings — no evidence in career text

### Root Cause

Skill counts (`retrieval_skill_count`, `vector_db_skill_count`) rewarded these profiles at face value. A Marketing Manager with expert FAISS scored identically to a genuine ML Engineer on the Technical Fit component.

### Honeypot Detection Architecture

Eight independent rule checks were implemented as pure functions:

```mermaid
flowchart TD
    HP1["SKILL_DURATION_INFLATION\ntotal_months > 6× yoe_months\nconf: 0.60-0.95"]
    HP2["EXPERT_WITH_ZERO_EXPERIENCE\nproficiency=expert + duration=0\nconf: 0.85-0.95"]
    HP3["NON_TECH_WITH_ADVANCED_AI_SKILLS\nnon-tech title + advanced AI skill\nconf: 0.55-0.88"]
    HP4["SUMMARY_TITLE_CONTRADICTION\ntechnical title + non-tech summary text\nconf: 0.50-0.78"]
    HP5["IMPOSSIBLE_TIMELINE\nend_date < start_date\nOR duration delta > 24mo\nconf: 0.65-0.92"]
    HP6["SUSPICIOUS_CAREER_PROGRESSION\nsenior ML title + all prior roles non-tech\nconf: 0.60-0.75"]
    HP7["TITLE_CAREER_MISMATCH\nML title + all roles in non-ML industries\nconf: 0.55-0.80"]
    HP8["DOMAIN_SKILL_CONTRADICTION\n>= 4 advanced NLP AND >= 4 advanced CV\nconf: 0.45-0.70"]

    HP1 & HP2 & HP3 & HP4 & HP5 & HP6 & HP7 & HP8 --> RULE{"max >= 0.85\nOR sum >= 1.60?"}
    RULE -->|Yes| FLAG["is_honeypot = True\nconfidence = max"]
    RULE -->|No| CLEAN["is_honeypot = False"]
```

### Before / After

```mermaid
graph LR
    subgraph BEFORE["Before Honeypot Detection"]
        B1["Marketing Manager\nexpert FAISS + Pinecone + RAG\n0 evidence in career text"] -->|Technical Fit: 7pts| HIGHBEFORE["Score: 55\nTop 100 eligible"]
    end

    subgraph AFTER["After Honeypot Detection"]
        A1["Marketing Manager\nexpert FAISS + Pinecone + RAG\n0 evidence in career text"] -->|NON_TECH_WITH_AI flag 0.88\nTITLE_CAREER_MISMATCH 0.75\nsum=1.63 >= 1.60| DECAY["Honeypot decay × 0.0\n+ Non-tech gate × 0.20\n+ Evidence floor × 0.25"]
        DECAY --> LOWAFTER["Score: ~0.5\nNever reaches Top 100"]
    end
```

---

## 5. Issue 4 — Evidence Floor Not Enforced

### Problem Observed

Candidates with high Career Fit and Availability scores (long tenure at product companies, fast notice period) but **zero retrieval/ranking evidence** in their career text were ranking in the Top 50. These were product engineers who had worked at ML companies but had never personally built retrieval systems.

### Root Cause

The initial scoring had no minimum requirement on evidence. A candidate with `evidence_score=5` (one mention of "semantic search" in passing) could compensate via high `career_fit` (5yr product ML) and `availability` (30d notice).

### Fix Applied

Three-tier evidence floor gate in `ScoringEngine.score()`:

```mermaid
flowchart LR
    E["evidence_score"] --> T1{"< 20?"}
    T1 -->|Yes| P1["score × 0.25\n75% penalty\nVirtually eliminates keyword-passers"]
    T1 -->|No| T2{"20-30?"}
    T2 -->|Yes| P2["score × 0.50\n50% penalty\nSignificant but recoverable"]
    T2 -->|No| PASS["No penalty\nevidence >= 30"]
```

**Why 30 is the threshold for "no penalty":** A candidate with just `retrieval_evidence=2 + ranking_evidence=2 + ml_production_evidence=1` scores `(2×5)+(2×5)+(1×4)=24` pts out of 30 = 80% evidence score — comfortably above both floors. This catches candidates who *only* mention retrieval concepts in passing without sustained career evidence.

---

## 6. Issue 5 — Activity Recency Blind Spot

### Problem Observed

After all gates were applied, several technically strong candidates (high evidence, product ML background) were ranking in the Top 20 but their `last_active_date` was 14–18 months ago. These candidates would be expensive to engage in practice.

### Root Cause

The initial behavioral score did not include `last_active_date`. A candidate inactive for 2 years scored the same on behavioral as an active candidate, as long as their `recruiter_response_rate` history was high.

### Fix Applied

Activity recency penalty in `_calc_behavioral()`:

```python
# Change 4: Activity recency penalty
if fv.days_since_active > 365:
    score -= 30.0   # > 1 year inactive
elif fv.days_since_active > 180:
    score -= 15.0   # 6–12 months inactive
```

**Threshold rationale (from dataset_profile.md):**
- 180 days = 6 months: JD explicitly states this as the "not hirable" boundary
- 365 days = 12 months: Out of market; would require extensive re-engagement campaign

---

## 7. Issue 6 — Reasoning Text Too Generic

### Problem Observed

Early reasoning texts looked like:
```
"AI Engineer with 6.3yrs; retrieval/semantic search; response rate 0.87."
```

The domain description used category labels ("retrieval/semantic search") rather than specific technology names that recruiters could verify against the profile. The format also did not include GitHub score as a standalone signal.

### Fix Applied

Updated `_build_reasoning_text()` to:
1. Use `top_domain_skills` (actual skill names from profile) as the domain segment
2. Add a **dedicated GitHub segment** when `github_activity_score >= 60`
3. Remove trailing period to match the specification format

**Before:**
```
"AI Engineer with 6.3yrs; retrieval/semantic search; github score 74."
```

**After:**
```
"AI Engineer with 6.3yrs; FAISS + Semantic Search; github 74; response rate 0.87"
```

---

## 8. Issue 7 — No Trust Signal for Borderline Profiles

### Problem Observed

Several profiles were not triggering honeypot detection (below the 0.85 threshold) but showed soft inconsistencies:
- A "Senior ML Engineer" claiming 16 years of experience expecting 6 LPA minimum salary (junior salary despite senior claim)
- Profiles with skill/YOE ratios of 4.5× (below the 6× hard trigger, but suspicious)
- Very sparse profiles (< 40% completeness) with high technical claims

These profiles were not being penalised at all, reaching the Top 40 on technical merit.

### Fix Applied

**Trust Layer** (`_extract_trust()`):

A soft multiplicative penalty (0.70–1.00) applied after honeypot decay, targeting three categories of inconsistency:

```mermaid
graph TD
    A["Profile enters Trust Layer"] --> B{Senior YOE >= 5\nExpecting < 8 LPA?}
    B -->|Yes| C["trust -= 0.10\nInflated experience claim"]
    B -->|No| D{Junior YOE < 2\nExpecting > 40 LPA?}
    C --> D
    D -->|Yes| E["trust -= 0.08\nInflated salary claim"]
    D -->|No| F{Skill months / YOE months\nin 4x-6x range?}
    E --> F
    F -->|Yes| G["trust -= sliding 0.0-0.10"]
    F -->|No| H{Profile < 40% complete?}
    G --> H
    H -->|Yes| I["trust -= 0.05"]
    H -->|No| J{Profile > 80% complete?}
    I --> J
    J -->|Yes| K["trust += 0.02 bonus"]
    J -->|No| L["clamp 0.70-1.00\nApply as score × trust"]
    K --> L
```

### Before / After

| Profile | Before | After |
|---|---|---|
| Senior 6yr, expecting 5 LPA, skill ratio 4.8× | No adjustment | trust=0.84; score × 0.84 |
| Sparse profile 35% complete, average signals | No adjustment | trust=0.95; score × 0.95 |
| Complete 85% profile, salary in range, ratio 2× | No adjustment | trust=1.0; no change (capped) |

---

## 9. Lessons Learned

| Lesson | Impact | Implementation |
|---|---|---|
| **Skill count alone is gameable** | Marketing Managers can list FAISS | Evidence score (25%) outweighs Technical Fit (15%) |
| **General ML keywords are too weak** | BAs describe company ML systems | Two-tier evidence model: strong vs general keywords |
| **Career text requires deliberate specificity** | Domain evidence keywords must be JD-specific | Only very specific terms (ndcg, bm25, ltr model) score evidence points |
| **Hard gates are necessary** | Soft scoring alone cannot exclude non-tech titles | Non-tech gate, evidence floor, technical eligibility gate |
| **Honeypot thresholds need dual conditions** | Single check can miss compound cases | `max >= 0.85 OR sum >= 1.60` dual rule |
| **Inactivity is an availability signal** | High skills + inactive 18 months = not hirable | `days_since_active` penalty in behavioral component |
| **Soft trust beats hard penalties for borderline cases** | Not every inconsistency is a hard disqualifier | Trust multiplier 0.70–1.00 for soft signals |
| **Reasoning must reference real data** | Generic text doesn't build recruiter trust | `top_domain_skills` uses actual technology names from profile |
| **Deterministic tiebreaking is required** | Non-deterministic ranking breaks reproducibility | 6-key sort tuple with candidate_id as ultimate tiebreaker |

---

## 10. Audit Results (Current State)

From `ranking_audit.md` generated by `audit.py`:

```mermaid
pie title Top 100 Title Category Distribution
    "ML_ENGINEER (37)" : 37
    "AI_ENGINEER (20)" : 20
    "SOFTWARE_ENGINEER (19)" : 19
    "DATA_SCIENTIST (10)" : 10
    "UNKNOWN (11)" : 11
    "DATA_ENGINEER (3)" : 3
```

| Metric | Before All Fixes | After All Fixes |
|---|---|---|
| NON_TECHNICAL titles in Top 100 | ~15–20 | **0** |
| MANAGER titles in Top 100 | ~5 | **0** |
| Honeypot confidence > 0.5 in Top 100 | ~8–12 | **0** |
| Product company coverage | ~70% | **100%** |
| Avg recruiter response rate | ~40% | **59.28%** |
