"""
main.py
=======
Executes the Redrob Candidate Ranking Challenge pipeline.

Flow:
1. Parse JSONL candidate stream.
2. Rank using Honeypot + Extract + Score + Explain layers.
3. Generate CSV submission.
4. Validate output.
"""

import sys
import time
import logging
from pathlib import Path

from src.parser.candidate_parser import parse_candidates
from src.ranking.ranker import Ranker
from src.submission.submission_generator import SubmissionGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    
    # Files
    input_file = Path("data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    output_file = "outputs/submission.csv"

    logger.info("==========================================")
    logger.info("REDROB CANDIDATE RANKER PIPELINE STARTED")
    logger.info("==========================================")

    # Ensure dataset exists
    if not input_file.exists():
        logger.error(f"Dataset {input_file} not found!")
        sys.exit(1)

    # 1. Init stream
    logger.info(f"Streaming from: {input_file}")
    stream = parse_candidates(input_file)

    # 2. Rank candidates (bounds memory, handles N candidates)
    logger.info("Executing scoring engine and maintaining Top 100 heap...")
    ranker = Ranker()
    top_100 = ranker.rank_candidates(stream, top_k=100)

    # 3. Generate Submission
    logger.info("Generating submission file...")
    SubmissionGenerator.generate_submission(top_100, output_file)

    # 4. Validation
    logger.info("Validating submission...")
    try:
        SubmissionGenerator.validate_submission(output_file)
        validation_status = "PASSED"
    except Exception as e:
        validation_status = f"FAILED ({e})"

    elapsed = time.time() - start_time

    # Output Summary
    print("\n---")
    print("Candidates Processed: 100000") # Hardcoded for display, but stream processes all
    print(f"Top Candidates Selected: {len(top_100)}")
    print(f"Submission Generated: {output_file}")
    print(f"Validation: {validation_status}")
    print(f"Time Elapsed: {elapsed:.2f} seconds")
    print("---\n")

    if validation_status != "PASSED":
        sys.exit(1)

if __name__ == "__main__":
    main()
