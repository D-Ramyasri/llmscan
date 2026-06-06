# semantic_engine.py
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a lightweight semantic model
try:
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    semantic_model = None

def analyze_semantics(prompt: str):
    """
    Returns semantic embedding or related info to understand the prompt intent.
    In a real app, this could compare against known bad prompt embeddings.
    """
    if semantic_model is None:
        return {"error": "Semantic model failed to load", "embedding": []}
    
    emb = semantic_model.encode(prompt)
    
    # Just returning some metadata for visualization
    return {
        "embedding_preview": emb[:5].tolist(),
        "vector_norm": float(np.linalg.norm(emb)),
        "status": "Semantic vector extracted successfully"
    }
