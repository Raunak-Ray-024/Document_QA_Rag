from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

# -----------------------------
# Engine Configuration
# -----------------------------
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # helps avoid stale connections
)

# -----------------------------
# Session Factory
# -----------------------------
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# -----------------------------
# Declarative Base
# -----------------------------
Base = declarative_base()


# -----------------------------
# DB Initialization
# -----------------------------
def init_db() -> None:
    """
    Initializes database:
    - Enables pgvector extension
    - Creates all tables from models
    """

    # Import inside function to prevent circular imports
    from app import models  # noqa: F401

    # Create pgvector extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create tables
    Base.metadata.create_all(bind=engine)


# -----------------------------
# Dependency
# -----------------------------
def get_db():
    """
    FastAPI dependency that provides a DB session
    and ensures proper cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


