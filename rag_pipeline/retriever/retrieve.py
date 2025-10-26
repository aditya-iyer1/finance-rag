# rag_pipeline/retriever/retrieve.py

import chromadb
from chromadb.config import Settings
from typing import List, Dict
from chromadb import PersistentClient
import os

def load_chroma_collection(persist_dir: str = "data/chroma_index"):
    # Convert to absolute path relative to project root
    if not os.path.isabs(persist_dir):
        # Get project root (2 levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        persist_dir = os.path.join(project_root, persist_dir)
    
    client = PersistentClient(path=persist_dir)
    return client.get_collection(name="finance_rag")


def query_chunks(
    query: str,
    top_k: int = 5,
    persist_dir: str = "data/chroma_index"
) -> List[Dict]:
    """
    Return top-k most relevant chunks for a given query.
    Safely handles cases where fewer results are returned.
    """
    collection = load_chroma_collection(persist_dir)

    # Perform query
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Handle empty or missing results gracefully
    if (
        not results 
        or "documents" not in results 
        or len(results["documents"]) == 0 
        or len(results["documents"][0]) == 0
    ):
        print(f"⚠️ No results found for query: '{query}'")
        return []

    # Only iterate through the actual number of returned results
    actual_k = len(results["documents"][0])
    top_chunks = []
    for i in range(actual_k):
        top_chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    print(f"✅ Retrieved {actual_k} relevant chunks (requested {top_k})")
    return top_chunks