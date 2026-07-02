# Changelog — Nitish v2 (Merged)

This changelog records the enhancements introduced in the Nitish v2 candidate ranking engine, incorporating the strongest search/retrieval heuristics and recruiter-facing signals while keeping the original architecture, weights, and pipeline structure fully intact.

---

## [2.0.0] - 2026-07-01

### Added
- **Central Config Module (`configs/skill_taxonomy.py`)**: Defines a compact 10-group hierarchical skill taxonomy (Retrieval, Dense Retrieval, Embeddings, Ranking, LLM Engineering, Serving, Evaluation, Recommendation Systems, Hybrid Search, RAG), 5-stage ideal retrieval pipeline sequence, production ML keywords, and experience quality keywords.
- **Richer Post-Ranking Audit (`outputs/top100_explainability_audit.md`)**: A tabular markdown audit for the Top 100 candidates providing candidate details, confidence category, trust score, JD match %, capability coverage ratio, technical strengths, weaknesses, and a concise summary of why they ranked.
- **Deterministic Test Runner (`run_tests_unittest.py`)**: A standard library based test runner to execute pytest-style assert tests inside the `unittest` framework without external dependencies.

### Changed
- **Advanced Composite Scoring Features**:
  - Expanded `FeatureVector` to include: `capability_score`, `pipeline_score`, `production_score`, `experience_quality_score`, and `confidence_category` with default values to ensure backward compatibility and prevent unit test breakage.
  - Implemented dynamic text and keyword coverage extraction in `candidate_features.py`.
- **Conservative Scorer Blending**:
  - Maintained the original 7 component weights exactly to avoid ranking regressions.
  - Blended `capability_score` (breadth) and `pipeline_score` (completeness bonus) directly into the `skill_score` computation.
  - Blended `experience_quality_score` (ownership/scale) and `production_score` (deployment/monitoring) directly into the `experience_score` computation.
  - Added a conditional fallback to use 100% original scores when composite metrics are unpopulated (e.g. in mock test objects).
- **Evidence-Based Reasoning (`src/reasoning/generator.py`)**:
  - Rewrote explanation generator to dynamically cite candidate-specific tools (e.g. BM25, FAISS, Pinecone, Elasticsearch) and factual metrics (YOE, MLOps, confidence).
  - Enforced a strict **180-character maximum** to produce concise, scannable recruiter-style entries.
  - Updated trust flag wording from "some profile attributes may benefit..." to `"Additional verification recommended."` to comply with safe trust scoring guidelines (Task 9).
- **Tie-Break Sorting Compliance**:
  - Fixed sorting logic in `builder.py` to sort descending by **rounded score** (to 4 decimal places) and tie-break ascending by `candidate_id` to ensure 100% compliance with the official validator.

### Fixed
- Fixed an `AttributeError` in reasoning generator when `profile` is `None` (for mock test candidates).
- Fixed `TypeError` in `test_scorer.py` and `test_reasoning.py` by adding default values to the new composite fields in `FeatureVector`.
