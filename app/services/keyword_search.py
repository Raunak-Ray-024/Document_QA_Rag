"""
Keyword Search Service - BM25 Algorithm for traditional keyword matching
This is NLP Task 1: Tokenization + Term Frequency
"""

from rank_bm25 import BM25Okapi
import nltk
from app.database import SessionLocal
from app.models import DocumentChunk, Document

# Download required NLTK data for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


def get_all_chunks_with_documents():
    """
    Fetch all chunks with their document names from database
    Returns: List of chunks with metadata
    """
    db = SessionLocal()
    results = db.query(DocumentChunk, Document.filename).join(
        Document, DocumentChunk.document_id == Document.id
    ).all()
    db.close()
    
    chunks_with_docs = []
    for chunk, filename in results:
        chunks_with_docs.append({
            "chunk_id": chunk.id,
            "chunk_text": chunk.chunk_text,
            "document_name": filename,
            "chunk_obj": chunk
        })
    
    return chunks_with_docs


def build_bm25_index():
    """
    Build BM25 index from all chunks
    BM25 requires tokenized text (words, not sentences)
    """
    chunks_data = get_all_chunks_with_documents()
    
    if not chunks_data:
        return None, []
    
    # Tokenize each chunk (convert to list of words)
    tokenized_chunks = []
    for chunk_data in chunks_data:
        # NLP Step: Tokenization - sentence to words
        tokens = nltk.word_tokenize(chunk_data["chunk_text"].lower())
        tokenized_chunks.append(tokens)
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)
    
    return bm25, chunks_data


def bm25_search(query: str, top_k: int = 10):
    """
    Search using BM25 keyword matching
    
    Args:
        query: User's question
        top_k: Number of results to return
    
    Returns:
        List of relevant chunks with BM25 scores
    """
    bm25, chunks_data = build_bm25_index()
    
    if not bm25 or not chunks_data:
        return []
    
    # NLP Step: Tokenize query
    tokenized_query = nltk.word_tokenize(query.lower())
    
    # Get BM25 scores for all chunks
    scores = bm25.get_scores(tokenized_query)
    
    # Get top_k indices
    top_indices = sorted(
        range(len(scores)), 
        key=lambda i: scores[i], 
        reverse=True
    )[:top_k]
    
    # Format results
    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include if there's SOME match
            results.append({
                "chunk_id": chunks_data[idx]["chunk_id"],
                "chunk_text": chunks_data[idx]["chunk_text"],
                "document_name": chunks_data[idx]["document_name"],
                "score": float(scores[idx]),
                "type": "bm25"
            })
    
    return results


# Test function
if __name__ == "__main__":
    print("Testing BM25 Search...")
    test_query = "machine learning"
    results = bm25_search(test_query, top_k=3)
    print(f"Query: '{test_query}'")
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  Score: {r['score']:.4f} - {r['chunk_text'][:100]}...")