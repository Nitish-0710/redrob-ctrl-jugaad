"""
src/reasoning/generator.py
==========================
Deterministically generates recruiter-style reasoning for candidate ranking.
Output is factual, concise (180-200 chars), citing specific candidate tools and metrics.
"""

from src.data.parser import Candidate
from src.features.candidate_features import FeatureVector
from src.ranking.scorer import CandidateScore

def generate_reasoning(candidate: Candidate, fv: FeatureVector, score: CandidateScore) -> str:
    """
    Generate a 100-180 character deterministic reasoning string based on the
    candidate's actual metrics, tools, and ranking scores.
    """
    exp = round(fv.implied_experience_years, 1)
    
    # 1. Identify specific tools matched in skills/career text
    skills_lower = {s.name.lower().strip() for s in candidate.skills} if candidate.skills else set()
    career_texts = []
    if candidate.profile:
        career_texts.extend([
            candidate.profile.headline,
            candidate.profile.summary,
            candidate.profile.current_title,
        ])
    if candidate.career_history:
        for job in candidate.career_history:
            career_texts.append(job.title)
            career_texts.append(job.description)
    combined_career_text = " ".join(filter(None, career_texts)).lower()

    # Tool mapping for reasoning display
    tool_map = {
        "elasticsearch": "Elasticsearch",
        "opensearch": "OpenSearch",
        "bm25": "BM25",
        "faiss": "FAISS",
        "pinecone": "Pinecone",
        "weaviate": "Weaviate",
        "milvus": "Milvus",
        "qdrant": "Qdrant",
        "pgvector": "pgvector",
        "lambdamart": "LambdaMART",
        "xgboost": "XGBoost",
        "langchain": "LangChain",
        "llamaindex": "LlamaIndex",
        "rag": "RAG",
        "lora": "LoRA",
        "peft": "PEFT"
    }

    matched_tools = []
    for key, val in tool_map.items():
        if key in skills_lower or key in combined_career_text:
            matched_tools.append(val)
            if len(matched_tools) >= 3: # limit to top 3 tools to save space
                break

    # 2. Identify core capability
    if fv.evidence_retrieval > 0 or fv.evidence_search > 0:
        if "FAISS" in matched_tools or "Pinecone" in matched_tools or "Weaviate" in matched_tools:
            primary_cap = "dense retrieval"
        elif "BM25" in matched_tools or "Elasticsearch" in matched_tools:
            primary_cap = "hybrid retrieval"
        else:
            primary_cap = "retrieval systems"
    elif fv.evidence_ranking > 0:
        primary_cap = "LTR ranking"
    elif fv.evidence_recommendation > 0:
        primary_cap = "recommendation systems"
    elif fv.evidence_nlp > 0:
        primary_cap = "NLP & LLM engineering"
    else:
        primary_cap = "applied ML engineering"

    # 3. Identify production maturity
    prod_phrases = []
    if fv.production_score >= 0.75:
        prod_phrases.append("prod scaling & MLOps")
    elif fv.production_score >= 0.5:
        prod_phrases.append("model deployment")
    elif fv.production_score >= 0.25:
        prod_phrases.append("production ownership")
    else:
        prod_phrases.append("applied ML")

    tool_str = f" using {', '.join(matched_tools)}" if matched_tools else ""
    prod_str = f", {prod_phrases[0]}" if prod_phrases else ""
    
    # Base reasoning
    reasoning = f"Designed {primary_cap}{tool_str}. {exp} yrs YOE{prod_str}. {score.confidence_category}."

    # 4. Handle trust flags / warnings safely (never imply fraud)
    concern = ""
    if fv.trust_score < 1.0:
        concern = " Additional verification recommended."
    elif fv.notice_period_days >= 60:
        concern = f" {fv.notice_period_days}-day notice period."
    elif fv.verification_score < 0.4:
        concern = " Limited verification signals."

    final_reasoning = (reasoning + concern).strip()
    
    # Strictly enforce 180 character limit (as requested)
    if len(final_reasoning) > 180:
        final_reasoning = final_reasoning[:177] + "..."
        
    return final_reasoning
