from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


# -----------------------------
# Question Request
# -----------------------------
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(5, ge=1, le=20)
    document_ids: Optional[List[int]] = None  # ← ADD THIS


    @validator("question")
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty or whitespace only")
        return v


# -----------------------------
# Source Reference
# -----------------------------
class SourceReference(BaseModel):
    chunk_text: str
    document_name: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)


# -----------------------------
# Answer Response
# -----------------------------
class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceReference]
    processing_time_ms: Optional[float] = None


# -----------------------------
# Document Upload Response
# -----------------------------
class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    chunks_created: int
    message: str
    status: str = "processing"

class DocumentInfo(BaseModel):
    """Information about an uploaded document"""
    id: int
    filename: str
    uploaded_at: datetime
    chunks_count: Optional[int] = None


# -----------------------------
# Health Response
# -----------------------------
class HealthResponse(BaseModel):
    status: str
    database: str
    version: str