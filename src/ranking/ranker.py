"""
src/ranking/ranker.py
=====================
Executes the full ranking pipeline for candidate streams and maintains 
a memory-efficient Top-K heap.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import Iterable, Any

from config.settings import DEFAULT_CONFIG, Config
from src.models.candidate import CandidateRecord
from src.scoring.honeypot_detector import HoneypotDetector
from src.scoring.feature_extractor import FeatureExtractor
from src.scoring.scoring_engine import ScoringEngine, ScoreCard
from src.explainability.reason_generator import ReasonGenerator, ExplanationResult

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    candidate_id: str
    final_score:  float
    rank:         int
    scorecard:    ScoreCard
    explanation:  ExplanationResult


class Ranker:
    """
    Coordinates the pipeline: parsing -> honeypot -> features -> scoring -> explainability.
    Uses a min-heap to keep memory strictly bounded to Top K elements.
    """

    def __init__(self, config: Config = DEFAULT_CONFIG) -> None:
        self.detector  = HoneypotDetector(config)
        self.extractor = FeatureExtractor(config)
        self.scorer    = ScoringEngine()
        self.explainer = ReasonGenerator()

    def rank_candidates(
        self, 
        candidate_stream: Iterable[CandidateRecord], 
        top_k: int = 100,
        log_every: int = 10_000
    ) -> list[RankedCandidate]:
        """
        Process a stream of candidates and return the Top K ranked.
        
        Memory bounds: O(K) memory regardless of stream size.
        """
        # Min-heap. We keep at most top_k elements.
        # Python's heapq is a min-heap, meaning heap[0] is the SMALLEST element.
        # So we push tuples of (score, tie_breakers..., candidate_data).
        # If heap size exceeds top_k, we pop the smallest, leaving the largest top_k.
        heap: list[tuple[Any, ...]] = []
        
        count = 0
        for candidate in candidate_stream:
            count += 1
            if count % log_every == 0:
                logger.info(f"Ranker processed {count} candidates...")

            # 1. Pipeline execution
            honeypot_res = self.detector.detect(candidate)
            features     = self.extractor.extract(candidate, honeypot_res)
            scorecard    = self.scorer.score(features)
            explanation  = self.explainer.generate(features, scorecard)

            # 2. Construct sorting tuple (primary score + tiebreakers)
            # All values must be comparable. Ties at the very bottom resolve by candidate_id string.
            sort_key = (
                scorecard.final_score,
                scorecard.evidence_score,
                scorecard.technical_fit_score,
                features.product_ml_experience_years,
                features.recruiter_response_rate,
                candidate.candidate_id,  # Ultimate tiebreaker ensures no tuple comparison crash
            )
            
            payload = (scorecard, explanation)
            entry   = (*sort_key, payload)

            # 3. Maintain top-k heap
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            else:
                # heappushpop pushes the new item and then pops and returns the smallest item.
                # This guarantees we only ever store top_k items.
                heapq.heappushpop(heap, entry)

        logger.info(f"Pipeline complete. Processed {count} total candidates.")

        # 4. Extract and sort descending
        # nlargest sorts descending natively.
        top_entries = heapq.nlargest(top_k, heap)

        # 5. Hydrate RankedCandidate dataclasses
        ranked_results = []
        for rank, entry in enumerate(top_entries, start=1):
            scorecard, explanation = entry[-1]
            ranked_results.append(
                RankedCandidate(
                    candidate_id=scorecard.candidate_id,
                    final_score=scorecard.final_score,
                    rank=rank,
                    scorecard=scorecard,
                    explanation=explanation
                )
            )

        return ranked_results
