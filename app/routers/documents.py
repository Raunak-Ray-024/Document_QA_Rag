"""
Documents Router - Handles PDF upload and document management endpoints.
"""

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models import Document, DocumentChunk
from app.schemas import DocumentUploadResponse, DocumentInfo
from app.services.ingestion import process_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF document for processing.
    
    - Validates file is PDF
    - Creates document record in database
    - Queues background processing (text extraction, chunking, embeddings)
    - Returns immediately with document ID
    
    Args:
        file: Uploaded PDF file (max size determined by FastAPI)
        
    Returns:
        DocumentUploadResponse with document_id and status
    """
    
    # 1. Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # 2. Read file content
    file_content = await file.read()
    
    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )
    
    # 3. Create document record in database
    document = Document(
        filename=file.filename,
        file_size=len(file_content)
    )
    db.add(document)
    db.commit()
    db.refresh(document)  # Get the auto-generated ID
    
    # 4. Queue background processing
    background_tasks.add_task(
        process_document,
        document_id=document.id,
        filename=file.filename,
        file_content=file_content
    )
    
    # 5. Return immediate response
    return DocumentUploadResponse(
        document_id=document.id,
        filename=file.filename,
        chunks_created=0,  # Unknown yet (processing in background)
        message="Document queued for processing",
        status="processing"
    )


@router.get("/", response_model=List[DocumentInfo])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all uploaded documents with their chunk counts.
    
    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
        
    Returns:
        List of DocumentInfo with document details and chunk counts
    """
    documents = db.query(Document).offset(skip).limit(limit).all()
    
    result = []
    for doc in documents:
        # Count chunks for this document
        chunk_count = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc.id
        ).count()
        
        result.append(DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            uploaded_at=doc.uploaded_at,
            chunks_count=chunk_count
        ))
    
    return result


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get details of a specific document.
    
    Args:
        document_id: ID of the document to retrieve
        
    Returns:
        DocumentInfo with document details and chunk count
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunk_count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).count()
    
    return DocumentInfo(
        id=document.id,
        filename=document.filename,
        uploaded_at=document.uploaded_at,
        chunks_count=chunk_count
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a document and all its chunks.
    
    Args:
        document_id: ID of the document to delete
        
    Returns:
        Deletion confirmation message
    """
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete all chunks first (cascade would work but explicit is clear)
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
    
    # Delete the document
    db.delete(document)
    db.commit()
    
    return {"message": f"Document {document_id} deleted successfully"}