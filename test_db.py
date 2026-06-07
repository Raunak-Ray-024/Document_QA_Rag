from app.database import SessionLocal
from app.models import DocumentChunk
from app.services.embeddings import generate_embedding
from sqlalchemy import text

db = SessionLocal()

# Generate a test embedding
test_embedding = generate_embedding('This is a test chunk')

# Create a test chunk
test_chunk = DocumentChunk(
    document_id=1,
    chunk_text='Test chunk text',
    chunk_index=0,
    embedding=test_embedding
)

# Add and commit
db.add(test_chunk)
db.commit()

# Verify it was stored
result = db.execute(text('SELECT id, chunk_text, embedding::text FROM document_chunks WHERE id = (SELECT MAX(id) FROM document_chunks)')).fetchone()
print(f'✅ Stored chunk ID: {result[0]}')
print(f'Chunk text: {result[1]}')
print(f'Embedding dimensions: {len(result[2].split(","))}')

db.close()