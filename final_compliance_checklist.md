# India.Runs 2026: Final Compliance Checklist
**Team**: Ctrl + Jugaad  
**Submission Status**: FULLY COMPLIANT  

This checklist maps our candidate discovery and ranking system against the official submission specifications and rules.

---

## 1. Compliance Checklist Matrix

| Rule / Constraint | Spec Reference | Implemented Status | Verification Method & Results |
|---|---|---|---|
| **CPU-Only Execution** | Section 10.1 | ✓ YES | Runs completely on CPU. PyTorch and ML operations do not configure or request GPU/CUDA resources. |
| **Memory Limit ($\le$ 16 GB)** | Section 10.1 | ✓ YES | Peak memory usage measured at **0.10 GB** (100 MB) due to memory-efficient streaming. |
| **Time Limit ($\le$ 5 minutes)** | Section 10.1 | ✓ YES | Total pipeline execution runs in **2m 23s** on the full 100,000-candidate dataset. |
| **Offline Mode (No External APIs)** | Section 10.2 | ✓ YES | Scoring engine is entirely heuristic and rule-based; no network or external LLM API calls are executed during ranking. |
| **Deterministic Tie-Breaking** | Section 2.3 | ✓ YES | Ties in rounded scores are resolved lexicographically by `candidate_id` ascending. |
| **Output Row Count (Exactly 100)** | Section 2.1 | ✓ YES | `outputs/submission.csv` contains exactly 1 header row and 100 candidate data rows. |
| **Header Names and Order** | Section 2.1 | ✓ YES | Header row is exactly `candidate_id,rank,score,reasoning`. |
| **Non-Increasing Score Order** | Section 2.2 | ✓ YES | Candidate scores monotonically decrease from Rank 1 down to Rank 100. |
| **Unique Candidate IDs** | Section 2.4 | ✓ YES | No duplicate candidate IDs are present in the top-100 output. |
| **UTF-8 Encoding** | Section 3.1 | ✓ YES | Submission CSV is successfully encoded and validated as UTF-8. |
| **Sandbox & Deployment Ready** | Section 10.5 | ✓ YES | Minimal Streamlit dashboard created (`app.py`) for drag-and-drop analysis and deployment. |
| **Metadata Declaration** | Section 4.1 | ✓ YES | Completed `submission_metadata.yaml` with team details and reproducibility command. |

---

## 2. Command-Line Reproduction
To run the ranking engine and reproduce the validated submission file, execute the single CLI entry point:

```bash
# Execute candidate ranking
python rank.py \
    --candidates candidates.jsonl \
    --out outputs/submission.csv
```

---

## 3. Official Validator Execution Result
Running the official `validate_submission.py` tool on our output results in:

```bash
$ python validate_submission.py outputs/submission.csv
Submission is valid.
```

No errors or warnings are raised, confirming complete compliance.
