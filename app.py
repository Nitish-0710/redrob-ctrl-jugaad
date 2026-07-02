import streamlit as st
import pandas as pd
import json
import tempfile
import os
import sys
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import stream
from src.data.validators import validate_candidate
from src.features.candidate_features import extract_features
from src.ranking.jd_parser import parse_jd
from src.ranking.scorer import score_candidate
from src.reasoning.generator import generate_reasoning
from configs.skill_taxonomy import CAPABILITY_GROUPS

st.set_page_config(
    page_title="Ctrl + Jugaad - Candidate Ranking Sandbox",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Redrob AI Challenge - Candidate Ranking Engine")
st.markdown("### Submission Sandbox — Developed by Team Ctrl + Jugaad")
st.markdown(
    "Upload a candidate profiles file in JSONL format (or a JSON array sample) to run the deterministic "
    "multi-dimensional candidate scoring and ranking pipeline."
)

uploaded_file = st.file_uploader("Upload candidates file (.jsonl or .json)", type=["jsonl", "json"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    suffix = ".json" if uploaded_file.name.endswith(".json") else ".jsonl"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=os.getcwd()) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)
        
    try:
        with st.spinner("Processing and ranking candidates..."):
            t0 = time.perf_counter()
            jd = parse_jd()
            
            # If the uploaded file is a JSON array, convert it to JSONL first
            if suffix == ".json":
                # Convert JSON list to temporary JSONL file
                with open(tmp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("JSON file must be a list of candidate profiles.")
                
                jsonl_path = tmp_path.with_suffix(".jsonl")
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for record in data:
                        f.write(json.dumps(record) + "\n")
                tmp_path = jsonl_path
            
            # Run ranking pipeline
            all_scores = []
            candidates_count = 0
            
            for c in stream(path=tmp_path, validate=False, skip_invalid=True):
                val = validate_candidate(c)
                fv = extract_features(c, val)
                score = score_candidate(fv, jd)
                all_scores.append((score.normalized_score, score.candidate_id))
                candidates_count += 1
                
            if candidates_count == 0:
                st.error("No valid candidate profiles parsed from the uploaded file.")
            else:
                # Sort descending
                all_scores.sort(key=lambda x: (-round(x[0], 4), x[1]))
                top_k = min(100, len(all_scores))
                top_k_ids = {x[1] for x in all_scores[:top_k]}
                
                # Fetch detailed data for top-K
                top_k_data = {}
                for c in stream(path=tmp_path, validate=False, skip_invalid=True):
                    if c.candidate_id in top_k_ids:
                        val = validate_candidate(c)
                        fv = extract_features(c, val)
                        score = score_candidate(fv, jd)
                        top_k_data[c.candidate_id] = (c, fv, score)
                        if len(top_k_data) == top_k:
                            break
                
                submission_rows = []
                display_rows = []
                
                for rank, (norm_score, cid) in enumerate(all_scores[:top_k], start=1):
                    c, fv, score = top_k_data[cid]
                    reasoning = generate_reasoning(c, fv, score)
                    
                    submission_rows.append({
                        "candidate_id": cid,
                        "rank": rank,
                        "score": round(norm_score, 4),
                        "reasoning": reasoning
                    })
                    
                    display_rows.append({
                        "Rank": rank,
                        "Candidate ID": cid,
                        "Current Title": c.profile.current_title,
                        "YoE (Implied)": round(fv.implied_experience_years, 1),
                        "AI Skills": fv.ai_skill_count,
                        "Trust Score": round(fv.trust_score, 2),
                        "Confidence": score.confidence_category,
                        "Final Score": round(norm_score, 4),
                        "Reasoning": reasoning
                    })
                    
                df_sub = pd.DataFrame(submission_rows)
                df_display = pd.DataFrame(display_rows)
                
                t1 = time.perf_counter()
                st.success(f"Successfully ranked {candidates_count:,} candidates in {t1-t0:.2f} seconds!")
                
                # Display Results
                st.subheader(f"🏆 Ranked Top {top_k} Candidates")
                st.dataframe(df_display, use_container_width=True)
                
                # Download Ranked CSV
                csv_data = df_sub.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Ranked CSV",
                    data=csv_data,
                    file_name="submission.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"An error occurred during pipeline execution: {e}")
        
    finally:
        # Cleanup temporary files
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass
