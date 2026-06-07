from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.database import init_db, get_db
from app.config import settings
from app.schemas import HealthResponse
from app.routers import documents, query  # ← Add 'documents' here

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-based Document Q&A System with FastAPI + pgvector",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Run when FastAPI starts"""
    print("🚀 Starting RAG System API...")
    print("🔧 Initializing database...")
    init_db()
    print("✅ Database initialized with pgvector support")
    print(f"📚 Using embedding model: {settings.embedding_model}")

# Include routers
app.include_router(documents.router)  # ← ADD THIS LINE
app.include_router(query.router)       # ← Already there (empty for now)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Document Q&A RAG System",
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /documents/upload",
            "list_documents": "GET /documents/",
            "get_document": "GET /documents/{document_id}",
            "delete_document": "DELETE /documents/{document_id}",
            "ask": "POST /query/ask (coming soon)"
        }
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Check if the API and database are working"""
    try:
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy",
        database=db_status,
        version=settings.app_version
    )