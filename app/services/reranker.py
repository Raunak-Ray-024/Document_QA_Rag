"""
Reranker Service - Uses Cross-Encoder for accurate relevance scoring
This is ADVANCED NLP: Cross-attention between query and chunk
"""

from sentence_transformers import CrossEncoder
import numpy as np

# Load cross-encoder model (downloads once, ~100MB)
# This model understands RELATIONSHIP between query and chunk
try:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✅ Cross-encoder model loaded")
except Exception as e:
    print(f"⚠️ Could not load cross-encoder: {e}")
    cross_encoder = None

"""
Reranker Service - Cross-encoder temporarily disabled due to memory issues
"""

# Comment out the cross-encoder loading
# from sentence_transformers import CrossEncoder
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

cross_encoder = None  # Disabled for now

def rerank_results(query: str, results: list, top_k: int = 3):
    """
    Rerank results - currently returns original results (no reranking)
    """
    if not results:
        return []
    
    # Just return top_k results without reranking
    return results[:top_k]
    
    # Prepare pairs for cross-encoder
    # Pair = (query, chunk_text) - model will compare them together
    pairs = [(query, r["chunk_text"]) for r in results]
    
    # Get relevance scores from cross-encoder
    # Higher score = more relevant to the query
    scores = cross_encoder.predict(pairs)
    
    # Add cross-encoder scores to results
    for i, r in enumerate(results):
        r["cross_encoder_score"] = float(scores[i])
        # Combine original score with cross-encoder score
        # Cross-encoder gets higher weight because it's more accurate
        r["final_score"] = (r["combined_score"] * 0.3) + (float(scores[i]) * 0.7)
    
    # Sort by final score and return top_k
    reranked = sorted(results, key=lambda x: x["final_score"], reverse=True)[:top_k]
    
    return reranked


# Test function (requires some results from hybrid search)
if __name__ == "__main__":
    print("Testing Reranker...")
    if cross_encoder:
        test_results = [
            {"chunk_text": "Machine learning is a subset of AI", "combined_score": 0.5},
            {"chunk_text": "Deep learning uses neural networks", "combined_score": 0.4},
        ]
        test_query = "What is machine learning?"
        reranked = rerank_results(test_query, test_results, top_k=2)
        print(f"Reranked {len(reranked)} results")
        for r in reranked:
            print(f"  Final score: {r['final_score']:.4f}")
    else:
        print("⚠️ Cross-encoder not available, skipping test")