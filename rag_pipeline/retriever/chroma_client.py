# rag_pipeline/retriever/chroma_client.py
"""
Unified ChromaDB client for consistent access across the RAG pipeline.
All ChromaDB operations should use get_collection() from this module to ensure
we're accessing the same backend and collection.
"""

import os
from chromadb import PersistentClient
from typing import Optional

# Default persist directory (relative to project root)
DEFAULT_PERSIST_DIR = "data/chroma_index"

# Collection name used throughout the pipeline
COLLECTION_NAME = "finance_rag"

# Cache the client to avoid creating multiple connections
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
    Uses the same backend configuration as load_chroma_collection().
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


def debug_chroma(persist_dir: str = DEFAULT_PERSIST_DIR):
    """
    Debug function to verify ChromaDB setup.
    Prints absolute persist path and list of collections.
    """
    resolved_dir = _resolve_persist_dir(persist_dir)
    client = get_client(persist_dir)
    collections = client.list_collections()
    
    print("=" * 60)
    print("ChromaDB Debug Info")
    print("=" * 60)
    print(f"Absolute persist directory: {resolved_dir}")
    print(f"Collections found: {[c.name for c in collections]}")
    print(f"Target collection: {COLLECTION_NAME}")
    
    if any(c.name == COLLECTION_NAME for c in collections):
        print(f"✅ Collection '{COLLECTION_NAME}' exists")
        collection = get_collection(persist_dir, COLLECTION_NAME)
        count = collection.count()
        print(f"   Chunk count: {count}")
    else:
        print(f"⚠️  Collection '{COLLECTION_NAME}' NOT FOUND")
        print(f"   Available collections: {[c.name for c in collections]}")
    print("=" * 60)
