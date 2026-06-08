"""
Ingestion Service - Handles PDF processing, chunking, and database storage.
No NLTK dependency - uses simple sentence splitting.
"""

import io
import re
from typing import List
from PyPDF2 import PdfReader

from app.database import SessionLocal
from app.models import Document, DocumentChunk
from app.services.embeddings import generate_embedding


def simple_sent_tokenize(text: str) -> List[str]:
    """
    Simple sentence tokenizer without NLTK.
    Splits on . ! ? followed by space or newline.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks using sentence boundaries."""
    if not text:
        return []
    
    # Split into sentences (simple method, no NLTK)
    sentences = simple_sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep overlap sentences
            overlap_text = ""
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) + 1 <= overlap:
                    overlap_text = s + " " + overlap_text
                    overlap_length += len(s) + 1
                else:
                    break
            
            if overlap_text:
                current_chunk = [overlap_text.strip()]
                current_length = len(overlap_text.strip())
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


async def process_document(document_id: int, filename: str, file_content: bytes):
    """Full ingestion pipeline for a single document."""
    print(f"📄 Processing document: {filename} (ID: {document_id})")
    
    text = extract_text_from_pdf(file_content)
    
    if not text:
        print(f"  ⚠️ No text extracted from {filename}")
        return
    
    print(f"  📝 Extracted {len(text)} characters")
    
    chunks = chunk_text(text)
    print(f"  🔢 Created {len(chunks)} chunks")
    
    db = SessionLocal()
    
    try:
        for idx, chunk in enumerate(chunks):
            print(f"  🧠 Generating embedding for chunk {idx + 1}/{len(chunks)}...")
            embedding = generate_embedding(chunk)
            
            chunk_record = DocumentChunk(
                document_id=document_id,
                chunk_text=chunk,
                chunk_index=idx,
                embedding=embedding
            )
            db.add(chunk_record)
        
        db.commit()
        print(f"  ✅ Successfully stored {len(chunks)} chunks for {filename}")
        
    except Exception as e:
        print(f"  ❌ Error processing {filename}: {e}")
        db.rollback()
        raise
    finally:
        db.close()