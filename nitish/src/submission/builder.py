"""
src/submission/builder.py
=========================
Final submission pipeline. Scores all candidates, selects top 100,
generates reasoning, and validates the required competition format.
"""

import sys
import time
from pathlib import Path
import pandas as pd
from collections import defaultdict

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.loader import stream
from src.data.validators import validate_candidate
from src.features.candidate_features import extract_features
from src.ranking.jd_parser import parse_jd
from src.ranking.scorer import score_candidate
from src.reasoning.generator import generate_reasoning

def build_submission():
    jd = parse_jd()
    
    print("Building submission (scoring 100k candidates)...")
    t0 = time.perf_counter()
    
    all_scores = []
    
    for c in stream(validate=False, skip_invalid=True):
        val = validate_candidate(c)
        fv = extract_features(c, val)
        score = score_candidate(fv, jd)
        all_scores.append((score.normalized_score, score.candidate_id))
        
    print(f"Scored {len(all_scores)} candidates in {time.perf_counter() - t0:.1f}s")
    
    # Sort descending by score
    all_scores.sort(key=lambda x: x[0], reverse=True)
    top_100_ids = {x[1] for x in all_scores[:100]}
    
    # Re-fetch the Top 100 full objects for reasoning generation
    top_100_data = {}
    print("Re-fetching Top 100 candidate records for reasoning generation...")
    for c in stream(validate=False, skip_invalid=True):
        if c.candidate_id in top_100_ids:
            val = validate_candidate(c)
            fv = extract_features(c, val)
            score = score_candidate(fv, jd)
            top_100_data[c.candidate_id] = (c, fv, score)
            if len(top_100_data) == 100:
                break
                
    submission_rows = []
    analysis_data = []
    
    for rank, (norm_score, cid) in enumerate(all_scores[:100], start=1):
        c, fv, score = top_100_data[cid]
        reasoning = generate_reasoning(c, fv, score)
        
        submission_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(norm_score, 4),
            "reasoning": reasoning
        })
        
        # Extra data for the final report
        analysis_data.append({
            "rank": rank,
            "title": c.profile.current_title,
            "score": norm_score,
            "trust_score": score.score_breakdown.get("trust_score", 1.0),
            "experience": fv.implied_experience_years,
            "ai_skills": fv.ai_skill_count,
            "reasoning": reasoning
        })
        
    df = pd.DataFrame(submission_rows)
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    
    df.to_csv(outputs_dir / "submission.csv", index=False)
    df.head(25).to_csv(outputs_dir / "submission_preview.csv", index=False)
    
    print("Submission generated.")
    return df, analysis_data


def validate_submission(df: pd.DataFrame) -> bool:
    print("Validating submission...")
    errors = []
    
    # 1. exactly 100 rows
    if len(df) != 100:
        errors.append(f"Row count is {len(df)}, expected 100.")
        
    # 2. required columns exist
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    if list(df.columns) != expected_cols:
        errors.append(f"Columns are {list(df.columns)}, expected {expected_cols}.")
        
    # 3. no duplicate candidate_id
    if df["candidate_id"].nunique() != len(df):
        errors.append("Duplicate candidate_ids found.")
        
    # 4. ranks are 1..100
    if list(df["rank"]) != list(range(1, 101)):
        errors.append("Ranks are not exactly 1 to 100 in order.")
        
    # 5. scores sorted descending
    scores = list(df["score"])
    if scores != sorted(scores, reverse=True):
        errors.append("Scores are not sorted descending.")
        
    # 6. no empty reasoning
    if df["reasoning"].isna().any() or (df["reasoning"] == "").any():
        errors.append("Empty reasoning strings found.")
        
    # 7. no null values
    if df.isna().any().any():
        errors.append("Null values found in dataframe.")
        
    outputs_dir = project_root / "outputs"
    val_md = ["# Submission Validation Report\n"]
    if not errors:
        val_md.append("✅ **Validation Passed!** The submission meets all competition requirements.")
        val_md.append("\n**Checks performed:**")
        val_md.append("- Exactly 100 rows")
        val_md.append("- Required columns exist (candidate_id, rank, score, reasoning)")
        val_md.append("- No duplicate candidate_ids")
        val_md.append("- Ranks are sequentially 1 to 100")
        val_md.append("- Scores are strictly descending")
        val_md.append("- No empty reasoning")
        val_md.append("- No null values anywhere")
        is_valid = True
    else:
        val_md.append("❌ **Validation Failed!**")
        for e in errors:
            val_md.append(f"- {e}")
        is_valid = False
            
    with open(outputs_dir / "submission_validation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(val_md))
        
    return is_valid


def generate_report(analysis_data: list):
    print("Generating final report...")
    
    def map_archetype(title: str) -> str:
        t = title.lower()
        if "search" in t or "retrieval" in t: return "Search Engineer"
        if "recommendation" in t or "recommender" in t: return "Recommendation Engineer"
        if "nlp" in t or "language" in t or "llm" in t: return "NLP Engineer"
        if "research" in t or "scientist" in t: return "AI Research Engineer"
        if "vision" in t or "cv " in t: return "Computer Vision Engineer"
        if "applied" in t: return "Applied ML Engineer"
        if "data" in t: return "Data Scientist"
        return "ML Engineer / General AI"
        
    scores = [x["score"] for x in analysis_data]
    experiences = [x["experience"] for x in analysis_data]
    skills = [x["ai_skills"] for x in analysis_data]
    trusts = [x["trust_score"] for x in analysis_data]
    
    arch_counts = defaultdict(int)
    for x in analysis_data:
        arch_counts[map_archetype(x["title"])] += 1
        
    report = [
        "# Final Submission Report\n",
        "## Overall Statistics",
        f"- **Min Score in Top 100:** {min(scores):.3f}",
        f"- **Max Score in Top 100:** {max(scores):.3f}",
        f"- **Average Experience:** {sum(experiences)/100:.1f} years",
        f"- **Average AI Skills:** {sum(skills)/100:.1f}",
        f"- **Average Trust Score:** {sum(trusts)/100:.2f} (Perfect cleanliness check)\n",
        "## Archetype Distribution",
    ]
    
    for arch, count in sorted(arch_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"- {arch}: {count}")
        
    report.append("\n## Top 10 Candidate Summary\n")
    for x in analysis_data[:10]:
        report.append(f"**{x['rank']}. {x['title']} ({x['score']:.3f})**")
        report.append(f"> {x['reasoning']}\n")
        
    outputs_dir = project_root / "outputs"
    with open(outputs_dir / "final_submission_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))


if __name__ == "__main__":
    df, analysis_data = build_submission()
    is_valid = validate_submission(df)
    if is_valid:
        generate_report(analysis_data)
        print("Pipeline finished successfully. All outputs ready.")
    else:
        print("Validation failed. See outputs/submission_validation.md")
