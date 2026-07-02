# India.Runs 2026: Repository Audit Report
**Date**: July 2, 2026  
**Status**: SUBMISSION READY  
**Auditor**: Senior Software Engineer & Hackathon Reviewer  

---

## 1. Executive Summary
This report details the final auditing pass on the **Intelligent Candidate Discovery & Ranking Engine** repository. The codebase has been audited for structural integrity, broken paths/imports, dependency configurations, and performance compliance. 

The evaluation confirms that the repository satisfies all formatting, structural, and performance constraints specified in the official Redrob Hackathon Guidelines.

---

## 2. Directory Layout & Folder Structure
Following the user constraint to preserve the current structure unless changes are required for reproducibility, the repository structure stands as:

* `configs/` — Contains configurations and taxonomies (Single Source of Truth).
* `src/` — Pipeline data loader, parser, validator, feature extractor, scorer, and reasoning generator.
* `tests/` — Full 15-test unit suite.
* `outputs/` — Folder containing `submission.csv`, preview csv, and diagnostic markdown logs.
* `rank.py` — The CLI reproducibility entry point.
* `app.py` — The Streamlit sandbox interface.
* `submission_metadata.yaml` — Declarations and team identity.
* `requirements.txt` — Package dependency lists.
* `merged/` & `nitish/` — Subfolders containing historical code and sandboxes. **These folders are ignored by git rules (using `.gitignore`) so they will not be committed to GitHub.**

---

## 3. GitHub Readiness Audit
* **Caches & Temp Caches**: `.gitignore` is updated to explicitly ignore `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, and `.streamlit/` folders.
* **Large Datasets**: `candidates.jsonl` (487 MB) is ignored to ensure compatibility with GitHub's 100 MB file limit.
* **Secrets Scan**: Scanned all codebase files (`configs/`, `src/`, `tests/`) for sensitive tokens, API credentials, or private keys. **No secrets or API keys are present.**
* **Virtual Environments**: `venv/` is fully ignored and excluded.
* **Intermediate Cache Files**: Parquet cache files in `data/processed/` and data directories (`data/raw/`, `data/interim/`) are ignored.

---

## 4. Portability & Import Integrity
* **Import Checks**: All internal modules import from the relative package root correctly. No absolute paths are hardcoded.
* **Path Resolution**: Fixed the path resolution logic in `configs/paths.py`. It dynamically falls back to the workspace root if `candidates.jsonl` is inside the root, preventing path breaks on clean clones.
* **Test Suite Verification**: Running `venv/Scripts/python.exe run_tests_unittest.py` results in a **100% pass rate** (15/15 tests passing successfully).

---

## 5. Dependency Audit
Dependencies declared in `requirements.txt` are verified for compatibility:
1. `numpy` and `pandas` — Used for candidate matrix representation and DataFrame exporting.
2. `pyarrow` — Used as an optimized backend for Parquet I/O.
3. `tqdm` — Streams progress loops.
4. `pytest` — Core testing framework.
5. `streamlit` and `psutil` — Added in Nitish v2 to run the submission sandbox and track peak RSS RAM.

---

## 6. Performance Audit Summary
The pipeline was ran using the absolute command line execution:
```bash
python rank.py --candidates candidates.jsonl --out outputs/submission.csv
```

### Metrics Profile
* **Runtime**: 2m 23s (well under the 5-minute limit)
* **Peak RSS RAM**: 0.10 GB (well under the 16 GB constraint due to O(1) streaming memory loaders)
* **Candidates Processed**: 99,999 profiles (plus 1 JSON parse error at line 48558)
* **CPU only**: Yes
* **External APIs**: No
* **Submission Valid**: Yes (verified using `validate_submission.py`)
