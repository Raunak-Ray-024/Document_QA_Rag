"""
Query Router - Handles Q&A endpoints with hybrid search + reranking
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import time

from app.database import get_db
from app.schemas import QuestionRequest, AnswerResponse, SourceReference
from app.services.hybrid_search import hybrid_search
from app.services.reranker import rerank_results

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    """
    Ask a question about your documents.
    
    This uses a 3-stage retrieval pipeline:
    1. Hybrid Search (BM25 keyword + Vector semantic)
    2. Cross-encoder Reranking (deep NLP)
    3. Return top-k relevant chunks
    
    Args:
        request: QuestionRequest with question and top_k
    
    Returns:
        AnswerResponse with relevant passages and sources
    """
    start_time = time.time()
    
    # Stage 1: Hybrid Search
    # Gets top results from both keyword and semantic search
    hybrid_results = hybrid_search(
        query=request.question,
        top_k=request.top_k * 2,  # Get more, then rerank
        keyword_weight=0.3,       # 30% importance to keywords
        vector_weight=0.7         # 70% importance to meaning
    )
    
    if not hybrid_results:
        return AnswerResponse(
            answer="No relevant documents found. Please upload some PDFs first.",
            sources=[],
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    # Stage 2: Rerank with Cross-Encoder
    # More accurate scoring by comparing query and chunk together
    reranked_results = rerank_results(
        query=request.question,
        results=hybrid_results,
        top_k=request.top_k
    )
    
    # Stage 3: Format sources for response
    sources = []
    for r in reranked_results:
        sources.append(SourceReference(
            chunk_text=r["chunk_text"][:500],  # Limit length
            document_name=r["document_name"],
            similarity_score=r["cross_encoder_score"] if "cross_encoder_score" in r else r["combined_score"]
        ))
    
    # Stage 4: Build user-friendly answer
    if len(sources) == 1:
        answer_text = f"Found 1 relevant passage from '{sources[0].document_name}':\n\n"
    else:
        answer_text = f"Found {len(sources)} relevant passages from your documents:\n\n"
    
    for i, src in enumerate(sources, 1):
        answer_text += f"{i}. From '{src.document_name}':\n"
        answer_text += f"   {src.chunk_text[:400]}..."
        if i < len(sources):
            answer_text += "\n\n"
    
    answer_text += "\n\n---\n💡 Tip: For a natural language answer, upgrade to LLM version."
    
    return AnswerResponse(
        answer=answer_text,
        sources=sources,
        processing_time_ms=(time.time() - start_time) * 1000
    )