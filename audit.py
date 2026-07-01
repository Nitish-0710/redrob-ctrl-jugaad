import csv
import json
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, ".")

from src.parser.candidate_parser import parse_candidates
from src.scoring.feature_extractor import FeatureExtractor, TitleCategory
from src.scoring.scoring_engine import ScoringEngine
from src.explainability.reason_generator import ReasonGenerator
from src.scoring.honeypot_detector import HoneypotDetector

def main():
    # Load top 100 IDs
    submission_path = Path("outputs/submission.csv")
    top_100_ids = {}
    with open(submission_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            top_100_ids[row["candidate_id"]] = {
                "rank": int(row["rank"]),
                "score": float(row["score"]),
                "reasoning": row["reasoning"]
            }

    # Stream to find the records
    input_file = Path("data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    candidates = {}
    
    gen = parse_candidates(input_file, progress_every=25000)
    for record in gen:
        if record.candidate_id in top_100_ids:
            candidates[record.candidate_id] = record
            if len(candidates) == 100:
                break

    # Extract features
    detector = HoneypotDetector()
    extractor = FeatureExtractor()
    scorer = ScoringEngine()
    explainer = ReasonGenerator()
    
    top_100_data = []
    
    for cid, meta in sorted(top_100_ids.items(), key=lambda x: x[1]["rank"]):
        record = candidates[cid]
        hp = detector.detect(record)
        fv = extractor.extract(record, hp)
        sc = scorer.score(fv)
        exp = explainer.generate(fv, sc)
        
        top_100_data.append({
            "candidate_id": cid,
            "rank": meta["rank"],
            "score": meta["score"],
            "current_title": fv.current_title,
            "years_experience": fv.years_experience,
            "title_category": fv.title_category,
            "honeypot_confidence": fv.honeypot_confidence,
            "strengths": exp.strengths,
            "concerns": exp.concerns,
            "product_company_ratio": fv.product_company_ratio,
            "product_company_count": fv.product_company_count,
            "service_company_count": fv.service_company_count,
            "recruiter_response_rate": fv.recruiter_response_rate,
            "github_activity_score": fv.github_activity_score,
            "notice_period_days": fv.notice_period_days,
            "interview_completion_rate": fv.interview_completion_rate,
            "cv_skill_count": fv.cv_skill_count,
            "retrieval_skill_count": fv.retrieval_skill_count,
            "evidence_score": sc.evidence_score
        })

    md = []
    md.append("# Ranking Audit: Top 100 Candidates")
    
    md.append("\n## Top 20 Candidate Review")
    for d in top_100_data[:20]:
        md.append(f"### Rank {d['rank']}: {d['candidate_id']}")
        md.append(f"- **Title**: {d['current_title']} ({d['title_category']})")
        md.append(f"- **Score**: {d['score']:.2f}")
        md.append(f"- **Experience**: {d['years_experience']} years")
        md.append(f"- **Honeypot Confidence**: {d['honeypot_confidence']:.2f}")
        md.append(f"- **Strengths**: {', '.join(d['strengths']) if d['strengths'] else 'None'}")
        md.append(f"- **Concerns**: {', '.join(d['concerns']) if d['concerns'] else 'None'}")
    
    md.append("\n## Title Distribution")
    titles = Counter(d['current_title'] for d in top_100_data)
    for title, count in titles.most_common(15):
        md.append(f"- {title}: {count}")
        
    md.append("\n## Title Category Distribution")
    categories = Counter(d['title_category'] for d in top_100_data)
    cat_keys = [c.value for c in TitleCategory]
    for cat in cat_keys:
        md.append(f"- {cat}: {categories.get(cat, 0)}")
        
    md.append("\n## Product Company Analysis")
    prod_exp = sum(1 for d in top_100_data if d['product_company_count'] > 0)
    service_only = sum(1 for d in top_100_data if d['product_company_count'] == 0 and d['service_company_count'] > 0)
    md.append(f"- Candidates with product company experience: {prod_exp}")
    md.append(f"- Candidates with ONLY service-company experience: {service_only}")
    
    md.append("\n## Behavioral Analysis (Averages for Top 100)")
    avg_resp = sum(d['recruiter_response_rate'] for d in top_100_data) / 100
    gh_cands = [d['github_activity_score'] for d in top_100_data if d['github_activity_score'] != -1]
    avg_gh = sum(gh_cands) / len(gh_cands) if gh_cands else 0
    avg_notice = sum(d['notice_period_days'] for d in top_100_data) / 100
    avg_inter = sum(d['interview_completion_rate'] for d in top_100_data) / 100
    
    md.append(f"- Recruiter Response Rate: {avg_resp:.2%}")
    md.append(f"- GitHub Activity Score: {avg_gh:.2f} (excluding non-linked)")
    md.append(f"- Notice Period Days: {avg_notice:.1f}")
    md.append(f"- Interview Completion Rate: {avg_inter:.2%}")
    
    md.append("\n## Honeypot Analysis")
    hps = [d for d in top_100_data if d['honeypot_confidence'] > 0.5]
    if hps:
        for hp in hps:
            md.append(f"- Rank {hp['rank']} ({hp['candidate_id']}): {hp['honeypot_confidence']:.2f}")
    else:
        md.append("None. Zero candidates in the Top 100 have honeypot confidence > 0.5.")
        
    md.append("\n## Red Flags")
    flags = []
    for d in top_100_data:
        if d['title_category'] in (TitleCategory.NON_TECHNICAL.value, TitleCategory.MANAGER.value):
            flags.append(f"- Rank {d['rank']} ({d['candidate_id']}): Non-technical/Manager title ({d['current_title']})")
        if d['cv_skill_count'] >= 4 and d['retrieval_skill_count'] == 0:
            flags.append(f"- Rank {d['rank']} ({d['candidate_id']}): CV-only candidate (CV skills: {d['cv_skill_count']}, Retrieval: 0)")
        if d['evidence_score'] < 30 and d['score'] > 60:
            flags.append(f"- Rank {d['rank']} ({d['candidate_id']}): High score ({d['score']}) but low evidence ({d['evidence_score']:.1f})")
            
    if flags:
        md.extend(flags)
    else:
        md.append("No major red flags detected. The ranking successfully filtered out keyword stuffers and non-technical titles.")
        
    md.append("\n## Final Assessment")
    if not flags and not hps and categories.get(TitleCategory.AI_ENGINEER.value, 0) > 0:
        md.append("Yes, this shortlist looks exceptionally credible. The ranking successfully pushed highly relevant AI/ML Engineers to the top, penalized honeypots completely (none in top 100), and favored candidates with real product-company ML experience over pure keyword stuffers or non-technical applicants.")
    else:
        md.append("The shortlist looks generally credible but requires some fine-tuning. There are some red flags (see above) which suggest certain features or penalties may need to be adjusted (e.g., strengthening the evidence score constraint or honeypot decay).")

    with open("ranking_audit.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    main()
