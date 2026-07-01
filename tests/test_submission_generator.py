"""
tests/test_submission_generator.py
==================================
Unit tests for CSV submission generation and strict validation rules.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking.ranker import RankedCandidate
from src.scoring.scoring_engine import ScoreCard
from src.explainability.reason_generator import ExplanationResult
from src.submission.submission_generator import SubmissionGenerator

def _make_candidate(rank: int, score: float, cid: str = None, reason: str = "Good") -> RankedCandidate:
    if cid is None:
        cid = f"CAND_{rank}"
    
    return RankedCandidate(
        candidate_id=cid,
        final_score=score,
        rank=rank,
        scorecard=ScoreCard(cid, 0,0,0,0,0,0,0,0,score),
        explanation=ExplanationResult(cid, "summary", [], [], "HIGH", reason)
    )

def test_generate_and_validate_success(tmp_path):
    """A valid top 100 list generates properly and passes validation."""
    out_file = tmp_path / "submission.csv"
    
    candidates = []
    # Generate 100 candidates with descending scores 100.0 -> 0.1
    for i in range(1, 101):
        candidates.append(_make_candidate(rank=i, score=100.0 - (i * 0.5)))
        
    SubmissionGenerator.generate_submission(candidates, out_file)
    
    assert out_file.exists()
    assert SubmissionGenerator.validate_submission(out_file) is True

def test_validate_missing_rows(tmp_path):
    out_file = tmp_path / "submission.csv"
    
    # Only 99 rows
    candidates = [_make_candidate(i, 100-i) for i in range(1, 100)]
    SubmissionGenerator.generate_submission(candidates, out_file)
    
    try:
        SubmissionGenerator.validate_submission(out_file)
        assert False, "Should raise ValueError for 99 rows"
    except ValueError as e:
        assert "exactly 100 rows" in str(e)

def test_validate_duplicate_ids(tmp_path):
    out_file = tmp_path / "submission.csv"
    
    candidates = [_make_candidate(i, 100-i) for i in range(1, 101)]
    candidates[5] = _make_candidate(6, 94, cid="CAND_1") # Duplicate ID
    
    SubmissionGenerator.generate_submission(candidates, out_file)
    try:
        SubmissionGenerator.validate_submission(out_file)
        assert False
    except ValueError as e:
        assert "Duplicate candidate_id" in str(e)

def test_validate_broken_rank_sequence(tmp_path):
    out_file = tmp_path / "submission.csv"
    
    candidates = [_make_candidate(i, 100-i) for i in range(1, 101)]
    candidates[5] = _make_candidate(7, 94) # Skips rank 6
    
    SubmissionGenerator.generate_submission(candidates, out_file)
    try:
        SubmissionGenerator.validate_submission(out_file)
        assert False
    except ValueError as e:
        assert "Rank sequence broken" in str(e)

def test_validate_non_descending_scores(tmp_path):
    out_file = tmp_path / "submission.csv"
    
    candidates = [_make_candidate(i, 100-i) for i in range(1, 101)]
    candidates[5] = _make_candidate(6, 99.0) # Score goes up
    
    SubmissionGenerator.generate_submission(candidates, out_file)
    try:
        SubmissionGenerator.validate_submission(out_file)
        assert False
    except ValueError as e:
        assert "Score sequence broken" in str(e)

def test_validate_reasoning_too_long(tmp_path):
    out_file = tmp_path / "submission.csv"
    
    candidates = [_make_candidate(i, 100-i) for i in range(1, 101)]
    long_reason = "A" * 301
    candidates[0] = _make_candidate(1, 100.0, reason=long_reason)
    
    SubmissionGenerator.generate_submission(candidates, out_file)
    try:
        SubmissionGenerator.validate_submission(out_file)
        assert False
    except ValueError as e:
        assert "too long" in str(e)

if __name__ == "__main__":
    # Create dummy tmp_path using pathlib
    import tempfile
    
    passed = failed = 0
    test_funcs = [
        test_generate_and_validate_success,
        test_validate_missing_rows,
        test_validate_duplicate_ids,
        test_validate_broken_rank_sequence,
        test_validate_non_descending_scores,
        test_validate_reasoning_too_long
    ]
    
    for f in test_funcs:
        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                f(Path(tmpdirname))
                print(f"  ✅ {f.__name__}")
                passed += 1
            except AssertionError as e:
                print(f"  ❌ {f.__name__}: {e}")
                failed += 1
            except Exception as e:
                print(f"  💥 {f.__name__}: {e}")
                failed += 1
                
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    import sys; sys.exit(0 if failed == 0 else 1)
