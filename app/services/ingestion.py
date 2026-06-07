"""
Ingestion Service - Handles PDF processing, chunking, and database storage.
This is the core pipeline that converts PDFs into searchable vector chunks.
"""

import nltk
import os

# Set NLTK data path to a writable directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required data (only if not already present)
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt.zip')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

import io
from typing import List
from PyPDF2 import PdfReader
import nltk
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Document, DocumentChunk
from app.services.embeddings import generate_embedding

# Download NLTK data for sentence tokenization (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        file_content: Raw bytes of the PDF file
        
    Returns:
        Combined text from all pages as a single string
    """
    reader = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Some pages might be empty
            text += page_text + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks using sentence boundaries.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk (default 1000)
        overlap: Characters to overlap between chunks (default 200)
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If adding this sentence exceeds chunk size and we have content
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            
            # Keep overlap: find how many sentences fit in overlap
            overlap_text = ""
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) + 1 <= overlap:
                    overlap_text = s + " " + overlap_text
                    overlap_length += len(s) + 1
                else:
                    break
            
            # Start new chunk with overlap sentences
            if overlap_text:
                current_chunk = [overlap_text.strip()]
                current_length = len(overlap_text.strip())
            else:
                current_chunk = []
                current_length = 0
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_length + 1  # +1 for space
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


async def process_document(document_id: int, filename: str, file_content: bytes):
    """
    Full ingestion pipeline for a single document.
    
    This function:
    1. Extracts text from PDF
    2. Splits into overlapping chunks
    3. Generates embeddings for each chunk
    4. Stores chunks + embeddings in database
    
    Args:
        document_id: ID of the document in the database
        filename: Original filename (for logging)
        file_content: Raw bytes of the PDF file
    """
    print(f"📄 Processing document: {filename} (ID: {document_id})")
    
    # Step 1: Extract text from PDF
    print("  📖 Extracting text from PDF...")
    text = extract_text_from_pdf(file_content)
    
    if not text:
        print(f"  ⚠️ No text extracted from {filename}")
        return
    
    print(f"  📝 Extracted {len(text)} characters")
    
    # Step 2: Split into chunks
    print("  ✂️ Chunking text...")
    chunks = chunk_text(text)
    print(f"  🔢 Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings and store in database
    db = SessionLocal()
    
    try:
        for idx, chunk in enumerate(chunks):
            print(f"  🧠 Generating embedding for chunk {idx + 1}/{len(chunks)}...")
            
            # Generate embedding vector
            embedding = generate_embedding(chunk)
            
            # Create database record
            chunk_record = DocumentChunk(
                document_id=document_id,
                chunk_text=chunk,
                chunk_index=idx,
                embedding=embedding
            )
            db.add(chunk_record)
        
        # Commit all chunks to database
        db.commit()
        print(f"  ✅ Successfully stored {len(chunks)} chunks for {filename}")
        
    except Exception as e:
        print(f"  ❌ Error processing {filename}: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# Optional: Test the functions individually
if __name__ == "__main__":
    # Test chunking
    test_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    chunks = chunk_text(test_text, chunk_size=50, overlap=20)
    print(f"Chunking test: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk[:50]}...")