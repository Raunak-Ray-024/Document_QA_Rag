from sentence_transformers import SentenceTransformer
import numpy as np

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize the model (downloads once, then runs locally)
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

def generate_embedding(text: str) -> list:
    """Generate embedding using local sentence-transformers"""
    embedding = model.encode(text)
    return embedding.tolist()

def generate_embeddings_batch(texts: list) -> list:
    """Generate embeddings for multiple texts"""
    embeddings = model.encode(texts)
    return embeddings.tolist()