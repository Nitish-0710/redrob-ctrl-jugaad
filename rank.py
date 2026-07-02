import argparse
import sys
import time
import os
import psutil
import threading
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import stream
from src.data.validators import validate_candidate
from src.features.candidate_features import extract_features
from src.ranking.jd_parser import parse_jd
from src.ranking.scorer import score_candidate
from src.reasoning.generator import generate_reasoning
from configs.skill_taxonomy import CAPABILITY_GROUPS

class PeakMemoryTracker:
    def __init__(self):
        self.peak_mem = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._track)
        
    def start(self):
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.thread.join()
        
    def _track(self):
        process = psutil.Process(os.getpid())
        while self.running:
            try:
                mem = process.memory_info().rss / (1024 ** 3)  # RSS in GB
                if mem > self.peak_mem:
                    self.peak_mem = mem
            except Exception:
                pass
            time.sleep(0.05)

def run_ranking_pipeline(candidates_path: Path, out_path: Path):
    tracker = PeakMemoryTracker()
    tracker.start()
    
    t0 = time.perf_counter()
    
    jd = parse_jd()
    
    all_scores = []
    candidates_count = 0
    
    # Custom stream using parameter path
    for c in stream(path=candidates_path, validate=False, skip_invalid=True):
        val = validate_candidate(c)
        fv = extract_features(c, val)
        score = score_candidate(fv, jd)
        all_scores.append((score.normalized_score, score.candidate_id))
        candidates_count += 1
        
    # Sort descending by rounded score, tie-break by candidate_id ascending
    all_scores.sort(key=lambda x: (-round(x[0], 4), x[1]))
    top_100_ids = {x[1] for x in all_scores[:100]}
    
    # Re-fetch the Top 100 full objects for reasoning generation
    top_100_data = {}
    for c in stream(path=candidates_path, validate=False, skip_invalid=True):
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
        
        # Technical strengths extraction
        skills_lower = {s.name.lower().strip() for s in c.skills}
        career_texts = [
            c.profile.headline,
            c.profile.summary,
            c.profile.current_title,
        ]
        for job in c.career_history:
            career_texts.append(job.title)
            career_texts.append(job.description)
        combined_career_text = " ".join(filter(None, career_texts)).lower()
        
        matched_groups = []
        for group_name, keywords in CAPABILITY_GROUPS.items():
            if any(kw in skills_lower or kw in combined_career_text for kw in keywords):
                matched_groups.append(group_name.replace("_", " ").title())
        strengths_str = ", ".join(matched_groups[:3]) if matched_groups else "General ML"
        
        # Weaknesses extraction
        weaknesses_list = []
        if fv.trust_score < 1.0:
            weaknesses_list.append("Trust audit recommended")
        if fv.notice_period_days > 60:
            weaknesses_list.append(f"{fv.notice_period_days}d notice")
        if fv.verification_score < 0.5:
            weaknesses_list.append("Few verifications")
        weaknesses_str = ", ".join(weaknesses_list) if weaknesses_list else "None"
        
        # Why Ranked extraction
        why_ranked_parts = []
        if score.score_breakdown.get("domain_score", 0) >= 0.5:
            why_ranked_parts.append("High domain alignment")
        if fv.capability_score >= 0.6:
            why_ranked_parts.append("Broad skill coverage")
        if fv.production_score >= 0.5:
            why_ranked_parts.append("Production ML maturity")
        why_ranked_str = " & ".join(why_ranked_parts) if why_ranked_parts else "Balanced score profile"
        
        analysis_data.append({
            "rank": rank,
            "candidate_id": cid,
            "title": c.profile.current_title,
            "score": norm_score,
            "trust_score": score.score_breakdown.get("trust_score", 1.0),
            "experience": fv.implied_experience_years,
            "ai_skills": fv.ai_skill_count,
            "reasoning": reasoning,
            "technical_strengths": strengths_str,
            "weaknesses": weaknesses_str,
            "why_ranked": why_ranked_str,
            "jd_match_pct": round(score.score_breakdown.get("domain_score", 0) * 100, 1),
            "capability_coverage": f"{len(matched_groups)}/{len(CAPABILITY_GROUPS)}",
            "confidence": score.confidence_category
        })
        
    df = pd.DataFrame(submission_rows)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_path, index=False)
    
    # Save preview next to out_path
    preview_path = out_path.parent / "submission_preview.csv"
    df.head(25).to_csv(preview_path, index=False)
    
    # Validate submission using built-in validate_submission helper
    from src.submission.builder import validate_submission, generate_report, generate_explainability_audit
    is_valid = validate_submission(df)
    
    if is_valid:
        generate_report(analysis_data)
        generate_explainability_audit(analysis_data)
        
    t1 = time.perf_counter()
    tracker.stop()
    
    elapsed = t1 - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    runtime_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    print("\n" + "="*40)
    print("RANKING PIPELINE COMPLETION SUMMARY")
    print("="*40)
    print(f"Runtime: {runtime_str}")
    print(f"Peak RAM: {tracker.peak_mem:.2f} GB")
    print(f"Candidates: {candidates_count:,}")
    print("CPU only: Yes")
    print("External APIs: No")
    print(f"Submission valid: {'Yes' if is_valid else 'No'}")
    print("="*40 + "\n")
    
    return is_valid, elapsed, tracker.peak_mem, candidates_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Candidate Discovery & Ranking Engine")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl", help="Path to raw candidates.jsonl dataset")
    parser.add_argument("--out", type=str, default="outputs/submission.csv", help="Path to output submission.csv file")
    
    args = parser.parse_args()
    
    c_path = Path(args.candidates)
    o_path = Path(args.out)
    
    if not c_path.exists():
        print(f"Error: Candidate file '{c_path}' does not exist.")
        sys.exit(1)
        
    is_valid, _, _, _ = run_ranking_pipeline(c_path, o_path)
    if not is_valid:
        print("Validation failed. Please check validation report.")
        sys.exit(1)
