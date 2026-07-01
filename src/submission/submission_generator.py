"""
src/submission/submission_generator.py
======================================
Generates and validates the final ranking CSV submission file.
"""

import csv
from pathlib import Path
from typing import Sequence

from src.ranking.ranker import RankedCandidate

class SubmissionGenerator:
    """Handles final CSV generation and strict validation."""

    @staticmethod
    def generate_submission(candidates: Sequence[RankedCandidate], output_path: str | Path) -> None:
        """
        Writes the RankedCandidates to a CSV file.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            
            for c in candidates:
                writer.writerow([
                    c.candidate_id,
                    c.rank,
                    f"{c.final_score / 100.0:.4f}",  # Change 5: normalize to 0–1 (sample_submission uses this range)
                    c.explanation.reasoning_text
                ])

    @staticmethod
    def validate_submission(output_path: str | Path) -> bool:
        """
        Strictly validates the generated CSV against submission requirements.
        Raises descriptive ValueError if any requirement fails.
        """
        path = Path(output_path)
        if not path.exists():
            raise ValueError(f"Submission file {path} does not exist.")

        rows = []
        with open(path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or set(reader.fieldnames) != {"candidate_id", "rank", "score", "reasoning"}:
                raise ValueError(f"Invalid headers: {reader.fieldnames}")
            for row in reader:
                rows.append(row)

        if len(rows) != 100:
            raise ValueError(f"Submission must have exactly 100 rows, found {len(rows)}")

        seen_ids = set()
        prev_rank = 0
        prev_score = float('inf')

        for i, row in enumerate(rows, start=1):
            c_id = row['candidate_id']
            rank = int(row['rank'])
            score = float(row['score'])
            reasoning = row['reasoning'].strip()

            # Unique IDs
            if c_id in seen_ids:
                raise ValueError(f"Duplicate candidate_id found at row {i}: {c_id}")
            seen_ids.add(c_id)

            # Ranks 1 to 100, sequential
            if rank != prev_rank + 1:
                raise ValueError(f"Rank sequence broken at row {i}: expected {prev_rank + 1}, got {rank}")
            prev_rank = rank

            # Descending scores
            if score > prev_score:
                raise ValueError(f"Score sequence broken at row {i}: score {score} is greater than previous {prev_score}")
            prev_score = score

            # Reasoning text
            if not reasoning:
                raise ValueError(f"Empty reasoning at row {i} for candidate {c_id}")
            if len(reasoning) > 300:
                raise ValueError(f"Reasoning text too long at row {i} ({len(reasoning)} > 300 chars)")

        return True
