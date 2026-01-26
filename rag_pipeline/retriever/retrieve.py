# rag_pipeline/retriever/retrieve.py

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rag_pipeline.retriever.chroma_client import get_collection

# Embedding model must match embed_store.py for consistent retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Cache the embedder to avoid reloading on every query
_embedder_cache = None

def get_embedder():
    """Get or create the embedding model (cached for efficiency)."""
    global _embedder_cache
    if _embedder_cache is None:
        _embedder_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"🔍 Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    return _embedder_cache

def load_chroma_collection(persist_dir: str = "data/chroma_index"):
    """
    Load ChromaDB collection using the unified client.
    DEPRECATED: Use get_collection() from chroma_client.py directly.
    Kept for backward compatibility.
    """
    return get_collection(persist_dir)


def query_chunks(
    query: str,
    top_k: int = 5,
    persist_dir: str = "data/chroma_index"
) -> List[Dict]:
    """
    Return top-k most relevant chunks for a given query.
    Uses hybrid retrieval (semantic search + keyword filtering) for better results.
    This combines the benefits of semantic similarity with keyword matching to ensure
    relevant chunks are found even when the embedding model doesn't capture the semantic
    relationship perfectly.
    """
    # Use hybrid retrieval which combines semantic and keyword matching
    from rag_pipeline.retriever.hybrid_retrieve import hybrid_retrieve
    
    chunks = hybrid_retrieve(query, k=top_k, persist_dir=persist_dir)
    
    if len(chunks) > 0:
        print(f"✅ Retrieved {len(chunks)} relevant chunks (requested {top_k})")
    else:
        print(f"⚠️ No results found for query: '{query}'")
    
    return chunks