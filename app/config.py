from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings using Pydantic for validation"""
    
    # OpenAI Configuration (optional now since we're using sentence-transformers)
    OPEN_API_KEY: str = ""
    
    # Database Configuration
    DATABASE_URL: str = ""
    
    # Application Configuration
    app_name: str = "Document Q&A RAG System"
    app_version: str = "1.0.0"
    
    # RAG Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Changed from OpenAI
    llm_model: str = "gpt-3.5-turbo"  # Will be used in Phase 3
    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_top_k: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from .env

# Create a single instance to import elsewhere
settings = Settings()