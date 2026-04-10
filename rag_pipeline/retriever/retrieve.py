# rag_pipeline/retriever/retrieve.py

import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Embedding model must match embed_store.py for consistent retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Cache the embedder to avoid reloading on every query
_embedder_cache = None

def get_embedder():
    """Get or create the embedding model (cached for efficiency)."""
    global _embedder_cache
    if _embedder_cache is None:
        _embedder_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL_NAME)
    return _embedder_cache


def query_chunks(
    query: str,
    top_k: int = 5,
    persist_dir: str = "data/chroma_index",
    debug: bool = False,
    active_doc_id: Optional[str] = None,
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
    from rag_pipeline.retriever.chroma_client import resolve_active_doc_id

    resolved_doc_id = resolve_active_doc_id(active_doc_id, persist_dir=persist_dir)
    
    chunks = hybrid_retrieve(
        query,
        k=top_k,
        persist_dir=persist_dir,
        debug=debug,
        active_doc_id=resolved_doc_id,
    )
    
    if len(chunks) > 0:
        logger.info(
            "Retrieved %d relevant chunks (requested %d) for doc_id=%s",
            len(chunks),
            top_k,
            resolved_doc_id,
        )
    else:
        logger.info("No results found for query: '%s' in doc_id=%s", query, resolved_doc_id)
    
    return chunks
