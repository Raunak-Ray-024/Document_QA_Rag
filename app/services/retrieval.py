"""
Vector Search Service - Semantic search using pgvector
This is NLP Task 2: Embeddings + Cosine Similarity
"""

from sqlalchemy import text
from app.database import SessionLocal
from app.services.embeddings import generate_embedding


def vector_search(query: str, top_k: int = 10):
    """
    Search using pgvector cosine similarity
    
    Args:
        query: User's question
        top_k: Number of results to return
    
    Returns:
        List of relevant chunks with similarity scores
    """
    db = SessionLocal()
    
    # NLP Step: Convert query to embedding (384 numbers)
    query_embedding = generate_embedding(query)
    
    # Convert Python list to pgvector string format
    embedding_str = f"[{','.join(map(str, query_embedding))}]"
    
    # Cosine similarity search using <=> operator
    # Lower distance = more similar
    results = db.execute(
        text("""
            SELECT 
                dc.id,
                dc.chunk_text,
                dc.document_id,
                d.filename,
                1 - (dc.embedding <=> :embedding) as similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            ORDER BY dc.embedding <=> :embedding
            LIMIT :top_k
        """),
        {"embedding": embedding_str, "top_k": top_k}
    ).fetchall()
    
    db.close()
    
    return [{
        "chunk_id": r[0],
        "chunk_text": r[1],
        "document_id": r[2],
        "document_name": r[3],
        "score": float(r[4]),  # similarity (0-1, higher = better)
        "type": "vector"
    } for r in results]


# Test function
if __name__ == "__main__":
    print("Testing Vector Search...")
    test_query = "machine learning"
    results = vector_search(test_query, top_k=3)
    print(f"Query: '{test_query}'")
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  Similarity: {r['score']:.4f} - {r['chunk_text'][:100]}...")