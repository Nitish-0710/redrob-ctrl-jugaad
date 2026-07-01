"""
src/evaluation/run_ranking_audit.py
===================================
Runs the full pipeline on 100k candidates to produce auditing CSVs and 
distribution metrics for the ranking engine.
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
from collections import Counter

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.loader import stream, LoadStats
from src.data.validators import validate_candidate
from src.features.candidate_features import extract_features
from src.ranking.jd_parser import parse_jd
from src.ranking.scorer import score_candidate, CandidateScore


def main():
    print("Loading JD...")
    jd = parse_jd()
    
    print("Streaming and scoring candidates...")
    t0 = time.perf_counter()
    
    scored_candidates = []
    # We will only keep full feature vectors for the top 500 to save memory,
    # but since memory isn't a huge issue, we can just keep feature dicts for all,
    # or just keep CandidateScore and re-extract for the top 100 later. 
    # Let's keep a simplified record for all.
    
    all_scores = []
    top_candidates_heap = [] # (not using heap, just list and sort later)
    
    features_cache = {}  # Store FV dicts only for top 100 to save RAM? 
                         # Actually, storing 100k FV dicts is ~100k * 60 floats ≈ 60MB. Totally fine.
    
    count = 0
    for candidate in stream(validate=False, skip_invalid=True):
        val_result = validate_candidate(candidate)
        fv = extract_features(candidate, val_result)
        score = score_candidate(fv, jd)
        
        all_scores.append(score)
        features_cache[score.candidate_id] = {
            "fv": fv.to_dict(),
            "candidate": candidate
        }
        
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} candidates...")
            
    print(f"Finished scoring {count} candidates in {time.perf_counter()-t0:.1f}s")
    
    # Sort
    print("Sorting...")
    all_scores.sort(key=lambda x: x.normalized_score, reverse=True)
    
    top_100_scores = all_scores[:100]
    
    # Generate top100_audit.csv
    print("Generating outputs/top100_audit.csv...")
    audit_rows = []
    for s in top_100_scores:
        fv_dict = features_cache[s.candidate_id]["fv"]
        c = features_cache[s.candidate_id]["candidate"]
        
        row = {
            "candidate_id": s.candidate_id,
            "current_title": c.profile.current_title,
            "years_experience": fv_dict["years_experience"],
            "ai_skill_count": fv_dict["ai_skill_count"],
            "domain_score": s.score_breakdown.get("domain_score", 0),
            "skill_score": s.score_breakdown.get("skill_score", 0),
            "experience_score": s.score_breakdown.get("experience_score", 0),
            "behavioral_score": s.score_breakdown.get("behavioral_score", 0),
            "verification_score": s.score_breakdown.get("verification_score", 0),
            "education_score": s.score_breakdown.get("education_score", 0),
            "availability_score": s.score_breakdown.get("availability_score", 0),
            "trust_score": s.score_breakdown.get("trust_score", 0),
            "final_score": s.normalized_score
        }
        audit_rows.append(row)
        
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    pd.DataFrame(audit_rows).to_csv(outputs_dir / "top100_audit.csv", index=False)
    
    # Generate top10_detailed.csv
    print("Generating outputs/top10_detailed.csv...")
    detailed_rows = []
    for s in all_scores[:10]:
        fv_dict = features_cache[s.candidate_id]["fv"]
        row = {"final_score": s.normalized_score}
        row.update(fv_dict)
        detailed_rows.append(row)
        
    pd.DataFrame(detailed_rows).to_csv(outputs_dir / "top10_detailed.csv", index=False)
    
    # Generate diagnostics info
    print("Calculating diagnostics...")
    metrics = {
        "score_distribution": {
            ">0.9": len([x for x in all_scores if x.normalized_score > 0.9]),
            "0.7-0.9": len([x for x in all_scores if 0.7 < x.normalized_score <= 0.9]),
            "0.5-0.7": len([x for x in all_scores if 0.5 < x.normalized_score <= 0.7]),
            "0.3-0.5": len([x for x in all_scores if 0.3 < x.normalized_score <= 0.5]),
            "<0.3": len([x for x in all_scores if x.normalized_score <= 0.3]),
        },
        "trust_distribution": {
            "1.0": len([x for x in all_scores if x.score_breakdown["trust_score"] == 1.0]),
            "<1.0": len([x for x in all_scores if x.score_breakdown["trust_score"] < 1.0]),
            "0.25": len([x for x in all_scores if x.score_breakdown["trust_score"] <= 0.25]),
        },
        "top100_ai_skill_counts": [features_cache[s.candidate_id]["fv"]["ai_skill_count"] for s in top_100_scores],
        "top100_experience": [features_cache[s.candidate_id]["fv"]["years_experience"] for s in top_100_scores],
        "top100_titles": Counter([features_cache[s.candidate_id]["candidate"].profile.current_title for s in top_100_scores]).most_common(10),
        "top100_domains_avg": sum(s.score_breakdown["domain_score"] for s in top_100_scores) / 100.0,
        "trust_penalized_top": []
    }
    
    # Find candidates who would have been top 100 based on raw_score but fell out due to trust
    all_scores_raw = sorted(all_scores, key=lambda x: x.raw_score, reverse=True)
    top_100_raw_ids = set(x.candidate_id for x in all_scores_raw[:100])
    top_100_final_ids = set(x.candidate_id for x in top_100_scores)
    
    fell_out = top_100_raw_ids - top_100_final_ids
    for i, s in enumerate(all_scores_raw):
        if s.candidate_id in fell_out and len(metrics["trust_penalized_top"]) < 5:
            metrics["trust_penalized_top"].append({
                "id": s.candidate_id,
                "raw_score": s.raw_score,
                "final_score": s.normalized_score,
                "trust": s.score_breakdown["trust_score"]
            })
            
    with open(outputs_dir / "diagnostics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("Done! Check outputs/")

if __name__ == "__main__":
    main()
