"""
Hybrid Search Service - Combines BM25 (keyword) and Vector (semantic) search
This combines traditional NLP with modern deep learning NLP
"""

from app.services.keyword_search import bm25_search
from app.services.retrieval import vector_search
from app.database import SessionLocal
from app.models import DocumentChunk


def get_chunk_by_id(chunk_id: int):
    """Helper to fetch a chunk object by ID"""
    db = SessionLocal()
    chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
    db.close()
    return chunk


def hybrid_search(
    query: str, 
    top_k: int = 5, 
    keyword_weight: float = 0.3, 
    vector_weight: float = 0.7
):
    """
    Combine BM25 and Vector search results using weighted scoring
    
    Args:
        query: User's question
        top_k: Number of results to return
        keyword_weight: Importance of keyword search (0-1)
        vector_weight: Importance of semantic search (0-1)
    
    Returns:
        Combined and sorted results from both search methods
    """
    
    # Get results from both search methods (get more for better merging)
    keyword_results = bm25_search(query, top_k=top_k * 2)
    vector_results = vector_search(query, top_k=top_k * 2)
    
    # Dictionary to combine results by chunk_id
    combined = {}
    
    # Add BM25 results
    if keyword_results:
        # Normalize BM25 scores to 0-1 range
        max_k_score = max(r["score"] for r in keyword_results)
        for r in keyword_results:
            chunk_id = r["chunk_id"]
            norm_score = r["score"] / max_k_score if max_k_score > 0 else 0
            combined[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk_text": r["chunk_text"],
                "document_name": r["document_name"],
                "bm25_score": norm_score,
                "vector_score": 0,
                "combined_score": norm_score * keyword_weight,
                "scores_detail": [f"bm25: {norm_score:.3f}"]
            }
    
    # Add Vector results (and merge with existing)
    if vector_results:
        for r in vector_results:
            chunk_id = r["chunk_id"]
            if chunk_id in combined:
                # Update existing entry
                combined[chunk_id]["vector_score"] = r["score"]
                combined[chunk_id]["combined_score"] += r["score"] * vector_weight
                combined[chunk_id]["scores_detail"].append(f"vector: {r['score']:.3f}")
            else:
                # New entry
                combined[chunk_id] = {
                    "chunk_id": chunk_id,
                    "chunk_text": r["chunk_text"],
                    "document_name": r["document_name"],
                    "bm25_score": 0,
                    "vector_score": r["score"],
                    "combined_score": r["score"] * vector_weight,
                    "scores_detail": [f"vector: {r['score']:.3f}"]
                }
    
    # Sort by combined score and return top_k
    sorted_results = sorted(
        combined.values(), 
        key=lambda x: x["combined_score"], 
        reverse=True
    )[:top_k]
    
    return sorted_results


# Test function
if __name__ == "__main__":
    print("Testing Hybrid Search...")
    test_query = "machine learning"
    results = hybrid_search(test_query, top_k=3)
    print(f"Query: '{test_query}'")
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  Combined Score: {r['combined_score']:.4f} - {r['scores_detail']}")
        print(f"    Text: {r['chunk_text'][:80]}...")