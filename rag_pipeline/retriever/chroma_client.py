# rag_pipeline/retriever/chroma_client.py
"""
Unified ChromaDB client for consistent access across the RAG pipeline.
All ChromaDB operations should use get_collection() from this module to ensure
we're accessing the same backend and collection.
"""

import logging
import os
from chromadb import PersistentClient
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default persist directory (relative to project root)
DEFAULT_PERSIST_DIR = "data/chroma_index"

# Collection name used throughout the pipeline
COLLECTION_NAME = "finance_rag"

# Cache the client to avoid creating multiple connections
_client_cache = None
_persist_dir_cache = None


def reset_client():
    """Reset the cached client. Call after deleting/recreating the index directory."""
    global _client_cache, _persist_dir_cache
    _client_cache = None
    _persist_dir_cache = None


def _get_project_root():
    """Get the project root directory (3 levels up from this file)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_persist_dir(persist_dir: str = DEFAULT_PERSIST_DIR) -> str:
    """
    Resolve persist directory to absolute path.
    Uses the same resolution logic as embed_store.py and retrieve.py.
    """
    if os.path.isabs(persist_dir):
        return persist_dir
    
    project_root = _get_project_root()
    return os.path.join(project_root, persist_dir)


def get_client(persist_dir: str = DEFAULT_PERSIST_DIR):
    """
    Get or create a PersistentClient (cached for efficiency).
    Uses the same backend configuration across the pipeline.
    """
    global _client_cache, _persist_dir_cache
    
    resolved_dir = _resolve_persist_dir(persist_dir)
    
    # Return cached client if same persist_dir
    if _client_cache is not None and _persist_dir_cache == resolved_dir:
        return _client_cache
    
    # Create new client
    _client_cache = PersistentClient(path=resolved_dir)
    _persist_dir_cache = resolved_dir
    
    return _client_cache


def get_collection(persist_dir: str = DEFAULT_PERSIST_DIR, collection_name: str = COLLECTION_NAME):
    """
    Get the ChromaDB collection using the unified client.
    This is the single source of truth for all ChromaDB access.
    
    Args:
        persist_dir: Persist directory (default: "data/chroma_index")
        collection_name: Collection name (default: "finance_rag")
    
    Returns:
        ChromaDB collection object
    """
    client = get_client(persist_dir)
    return client.get_collection(name=collection_name)


def list_indexed_doc_ids(persist_dir: str = DEFAULT_PERSIST_DIR) -> List[str]:
    """Return sorted unique doc_ids currently present in the collection."""
    try:
        collection = get_collection(persist_dir, COLLECTION_NAME)
        all_data = collection.get(include=["metadatas"])
    except Exception:
        return []

    doc_ids = set()
    for metadata in all_data.get("metadatas", []):
        if isinstance(metadata, dict) and metadata.get("doc_id"):
            doc_ids.add(metadata["doc_id"])
    return sorted(doc_ids)


def delete_doc_id(doc_id: str, persist_dir: str = DEFAULT_PERSIST_DIR) -> int:
    """Delete all chunks for a specific doc_id. Returns deleted count when available."""
    collection = get_collection(persist_dir, COLLECTION_NAME)
    before_count = collection.count()
    collection.delete(where={"doc_id": doc_id})
    after_count = collection.count()
    return max(before_count - after_count, 0)


def resolve_active_doc_id(active_doc_id: Optional[str], persist_dir: str = DEFAULT_PERSIST_DIR) -> str:
    """
    Resolve the active document for querying.

    - If one indexed document exists, it becomes the implicit default.
    - If multiple documents exist, caller must specify active_doc_id.
    - If active_doc_id is specified, it must exist in the index.
    """
    indexed_doc_ids = list_indexed_doc_ids(persist_dir)
    if not indexed_doc_ids:
        raise ValueError("No indexed documents found. Run embed_chunks_cli.py first.")

    if active_doc_id:
        if active_doc_id not in indexed_doc_ids:
            raise ValueError(
                f"Document '{active_doc_id}' is not indexed. Available documents: {', '.join(indexed_doc_ids)}"
            )
        return active_doc_id

    if len(indexed_doc_ids) == 1:
        return indexed_doc_ids[0]

    raise ValueError(
        "Multiple documents are indexed. Specify active_doc_id to avoid mixing filings. "
        f"Available documents: {', '.join(indexed_doc_ids)}"
    )


def is_single_doc_mode(persist_dir: str = DEFAULT_PERSIST_DIR) -> bool:
    """
    Detect if we're in single-document mode (only one doc_id in the collection).
    In single-doc mode, entity terms should be downweighted.
    """
    try:
        return len(list_indexed_doc_ids(persist_dir)) <= 1
    except Exception:
        return True  # Default to single-doc on error


def debug_chroma(persist_dir: str = DEFAULT_PERSIST_DIR):
    """
    Debug function to verify ChromaDB setup.
    Prints absolute persist path and list of collections.
    """
    resolved_dir = _resolve_persist_dir(persist_dir)
    client = get_client(persist_dir)
    collections = client.list_collections()
    
    logger.debug("=" * 60)
    logger.debug("ChromaDB Debug Info")
    logger.debug("=" * 60)
    logger.debug("Absolute persist directory: %s", resolved_dir)
    logger.debug("Collections found: %s", [c.name for c in collections])
    logger.debug("Target collection: %s", COLLECTION_NAME)
    
    if any(c.name == COLLECTION_NAME for c in collections):
        logger.debug("Collection '%s' exists", COLLECTION_NAME)
        collection = get_collection(persist_dir, COLLECTION_NAME)
        count = collection.count()
        logger.debug("   Chunk count: %d", count)
        logger.debug("   Single-doc mode: %s", is_single_doc_mode(persist_dir))
    else:
        logger.debug("Collection '%s' NOT FOUND", COLLECTION_NAME)
        logger.debug("   Available collections: %s", [c.name for c in collections])
    logger.debug("=" * 60)
