"""
configs/skill_taxonomy.py
=========================
Hierarchical skill taxonomy, pipeline completeness definitions, and keyword lists 
for advanced capability, production ML, and experience quality scoring.
"""

from typing import Dict, Set, List

# Compact capability groups (around 10 groups as requested)
CAPABILITY_GROUPS: Dict[str, Set[str]] = {
    "retrieval": {
        "bm25", "elasticsearch", "opensearch", "solr", "lucene", "retrieval", "information retrieval"
    },
    "dense_retrieval": {
        "faiss", "pinecone", "weaviate", "milvus", "pgvector", "qdrant", "annoy", "hnsw", "approximate nearest neighbor"
    },
    "embeddings": {
        "sentence transformers", "bge", "e5", "minilm", "mpnet", "word2vec", "bert embeddings", "embeddings"
    },
    "ranking": {
        "learning to rank", "lambdamart", "xgboost ranker", "ranknet", "pairwise ranking", "listwise ranking", "reranking", "re-ranking"
    },
    "llm_engineering": {
        "lora", "qlora", "peft", "huggingface", "transformers", "fine-tuning", "prompt engineering", "llm"
    },
    "serving": {
        "bentoml", "torchserve", "triton", "fastapi", "flask", "model serving", "inference"
    },
    "evaluation": {
        "ndcg", "recall@k", "mrr", "precision@k", "offline evaluation", "online evaluation", "a/b testing"
    },
    "recommendation_systems": {
        "collaborative filtering", "matrix factorization", "two-tower", "deepfm", "recommendation", "recommender"
    },
    "hybrid_search": {
        "hybrid search", "reciprocal rank fusion", "rrf", "hybrid retrieval"
    },
    "rag": {
        "rag", "retrieval augmented generation", "langchain", "llamaindex"
    }
}

# Ideal pipeline stages in sequence (used for calculating pipeline completeness bonus)
PIPELINE_STAGES: List[str] = [
    "embeddings",       # Stage 1: Representation
    "dense_retrieval",  # Stage 2: Vector search
    "hybrid_search",    # Stage 3: Combination
    "ranking",          # Stage 4: Re-ranking
    "evaluation"        # Stage 5: Evaluation
]

# Keywords indicating production ML maturity
PRODUCTION_ML_KEYWORDS: Dict[str, Set[str]] = {
    "deployment": {
        "deploy", "deployment", "production", "kubernetes", "docker", "mlops", "ci/cd", "airflow", "kubeflow", "sagemaker"
    },
    "monitoring": {
        "monitor", "monitoring", "drift", "mlflow", "prometheus", "grafana", "drift detection", "data drift", "alerting"
    },
    "scaling": {
        "scale", "scaling", "high throughput", "distributed", "spark", "ray", "sharding", "replication"
    },
    "latency": {
        "latency", "microsecond", "millisecond", "onnx", "tensorrt", "quantization", "quantize", "throughput optimization"
    }
}

# Keywords indicating high experience quality (ownership, scale, leadership, impact)
EXPERIENCE_QUALITY_KEYWORDS: Dict[str, Set[str]] = {
    "ownership": {
        "owned", "architected", "designed", "led", "built", "production ownership", "end-to-end", "spearheaded"
    },
    "scale": {
        "millions of users", "scale", "high-throughput", "large-scale", "petabyte", "terabyte", "10m+", "100m+", "rps"
    },
    "leadership": {
        "mentored", "led", "tech lead", "principal", "staff", "manager", "guidance", "supervise", "supervised"
    },
    "impact": {
        "improved by", "increased", "reduced latency", "optimized", "optimization impact", "kpi", "saved", "accelerated"
    }
}
