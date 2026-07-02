"""
src/reasoning/generate_examples.py
==================================
Generates reasoning strings for the Top 10 candidates and writes them to markdown.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.loader import stream
from src.data.validators import validate_candidate
from src.features.candidate_features import extract_features
from src.ranking.jd_parser import parse_jd
from src.ranking.scorer import score_candidate
from src.reasoning.generator import generate_reasoning

def main():
    jd = parse_jd()
    print("Scoring all candidates to find Top 10...")
    all_scored = []
    candidates = {}
    fvs = {}
    
    t0 = time.perf_counter()
    for c in stream(validate=False, skip_invalid=True):
        val = validate_candidate(c)
        fv = extract_features(c, val)
        score = score_candidate(fv, jd)
        
        # Only keep top N in memory, using a simple sort later.
        # To avoid keeping 100k objects, we just store (score, cid)
        all_scored.append((score.normalized_score, c.candidate_id, score, fv, c))
        
    print(f"Scored {len(all_scored)} in {time.perf_counter()-t0:.1f}s")
    
    # Sort by normalized_score desc
    all_scored.sort(key=lambda x: x[0], reverse=True)
    top_20 = all_scored[:20]
    
    md_lines = ["# Final Reasoning Examples (Top 20 Candidates)\n"]
    
    for rank, (norm_score, cid, score, fv, c) in enumerate(top_20, start=1):
        reasoning = generate_reasoning(c, fv, score)
        md_lines.append(f"### Rank {rank}: {c.profile.current_title} ({cid})")
        md_lines.append(f"**Score:** {norm_score:.3f}")
        md_lines.append(f"**Reasoning:** {reasoning}\n")
        md_lines.append(f"*(Character count: {len(reasoning)})*\n")
        md_lines.append("---")
        
    outputs_dir = project_root / "outputs"
    out_path = outputs_dir / "reasoning_examples_final.md"
    with open(out_path, "w") as f:
        f.write("\n".join(md_lines))
        
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
