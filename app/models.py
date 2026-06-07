from sqlalchemy import Column, Integer, String, Text, DateTime, Index, ForeignKey, func
from pgvector.sqlalchemy import Vector

from app.database import Base


# -----------------------------
# Document Model
# -----------------------------
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, server_default=func.now())


# -----------------------------
# Document Chunk Model
# -----------------------------
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)

    document_id = Column(Integer, nullable=False, index=True)

    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)

    embedding = Column(Vector(384), nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    # -----------------------------
    # Indexes
    # -----------------------------
    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_ivfflat",
            "embedding",
            postgresql_using="ivfflat",
        ),
    )