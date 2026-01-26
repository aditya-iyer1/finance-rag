# rag_pipeline/retriever/hybrid_retrieve.py

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rag_pipeline.retriever.chroma_client import get_collection
from rag_pipeline.retriever.retrieve import get_embedder

# Embedding model must match ingestion (same as retrieve.py)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"


# In keyword_filter(), fix keyword extraction to strip punctuation:

def keyword_filter(chunks: List[Dict], query: str) -> List[Dict]:
    """Filter chunks that contain query keywords in text or metadata."""
    import re
    # Extract meaningful keywords (words longer than 3 chars, excluding common stopwords)
    query_lower = query.lower()
    # Strip punctuation from query before splitting
    query_clean = re.sub(r'[^\w\s]', ' ', query_lower)
    stopwords = {'where', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [w for w in query_clean.split() if len(w) > 3 and w not in stopwords]
    
    # If no good keywords, use all words longer than 2 chars
    if not keywords:
        keywords = [w for w in query_clean.split() if len(w) > 2]

    print(f"DEBUG: keywords={keywords}")
    filtered = []
    for chunk in chunks:
        text_lower = chunk.get("text", "").lower()
        metadata = chunk.get("metadata", {})
        metadata_str = " ".join(str(v).lower() for v in metadata.values())
        
        # Count how many keywords match (prioritize chunks with more keyword matches)
        keyword_matches = sum(1 for kw in keywords if kw in text_lower or kw in metadata_str)
        
        # Only include chunks that match at least one keyword
        if keyword_matches > 0:
            filtered.append((chunk, keyword_matches))
    
    # Sort by number of keyword matches (descending) to prioritize better matches
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the chunks (without match counts)
    return [chunk for chunk, _ in filtered]


def hybrid_retrieve(query: str, k: int = 5, persist_dir: str = "data/chroma_index") -> List[Dict]:
    """
    Hybrid retrieval: semantic search + keyword filtering.
    Uses unified ChromaDB client to ensure consistent access to finance_rag collection.
    """
    collection = get_collection(persist_dir)
    
    # Use same embedding model as ingestion
    embedder = get_embedder()
    query_embedding = embedder.encode([query])[0].tolist()
    
    # Semantic retrieval: get more candidates for recall
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k * 2,  # Get more for recall
        include=["documents", "metadatas", "distances"]
    )
    
    # Handle empty results
    if (
        not semantic_results 
        or "documents" not in semantic_results 
        or len(semantic_results["documents"]) == 0 
        or len(semantic_results["documents"][0]) == 0
    ):
        return []
    
    # Convert to dict format
    semantic_chunks = []
    if semantic_results and "documents" in semantic_results and len(semantic_results["documents"][0]) > 0:
        for i in range(len(semantic_results["documents"][0])):
            metadata = semantic_results["metadatas"][0][i] if semantic_results["metadatas"] and len(semantic_results["metadatas"][0]) > i else {}
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Ensure required metadata fields exist
            if "doc_id" not in metadata:
                metadata["doc_id"] = "unknown"
            if "chunk_id" not in metadata:
                metadata["chunk_id"] = "unknown"
            if "section" not in metadata:
                metadata["section"] = "unknown"
            
            semantic_chunks.append({
                "text": semantic_results["documents"][0][i],
                "metadata": metadata,
                "distance": semantic_results["distances"][0][i] if semantic_results["distances"] and len(semantic_results["distances"][0]) > i else None
            })
    
    # Check if top semantic result has poor match (high distance)
    top_distance = semantic_chunks[0].get("distance") if semantic_chunks else None
    use_keyword_fallback = top_distance and top_distance > 15.0

    # Keyword filtering on semantic results
    keyword_matches_semantic = keyword_filter(semantic_chunks, query)

    # If semantic results are poor, do keyword search across ALL chunks
    # (regardless of whether semantic keyword matches exist, since they might not be relevant)
    if use_keyword_fallback:
        all_data = collection.get(include=["documents", "metadatas"])
        all_chunks = []
        for i, doc in enumerate(all_data["documents"]):
            metadata = all_data["metadatas"][i] if all_data["metadatas"] else {}
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Ensure required fields
            if "doc_id" not in metadata:
                metadata["doc_id"] = "unknown"
            if "chunk_id" not in metadata:
                metadata["chunk_id"] = "unknown"
            if "section" not in metadata:
                metadata["section"] = "unknown"
            
            all_chunks.append({
                "text": doc,
                "metadata": metadata,
                "distance": None
            })
        
        # Filter all chunks by keywords
        keyword_matches_full = keyword_filter(all_chunks, query)
        
        # Debug output
        print(f"DEBUG: use_keyword_fallback={use_keyword_fallback}, keyword_matches_full count={len(keyword_matches_full)}")
        for i, km in enumerate(keyword_matches_full[:3]):
            print(f"  Keyword match {i+1}: {km['metadata'].get('section', 'unknown')} - {km['text'][:100]}...")
        
        # Use full DB keyword matches (they're more comprehensive)
        keyword_matches = keyword_matches_full
    else:
        # Use semantic keyword matches only
        keyword_matches = keyword_matches_semantic
    
    # Combine with deduplication (by text content)
    # If we have keyword matches from full DB search, prioritize them over poor semantic results
    if use_keyword_fallback and len(keyword_matches) > 0:
        # Prioritize keyword matches when semantic results are poor
        unique_chunks = {}
        # Add keyword matches first (they're more relevant)
        for chunk in keyword_matches:
            text = chunk.get("text", "")
            if text not in unique_chunks:
                unique_chunks[text] = chunk
        
        # Then add semantic results as fallback (only if we don't have enough keyword matches)
        for chunk in semantic_chunks:
            text = chunk.get("text", "")
            if text not in unique_chunks and len(unique_chunks) < k:
                unique_chunks[text] = chunk
    else:
        # Normal case: prioritize semantic results, add keyword matches as supplement
        unique_chunks = {}
        # Prioritize semantic results first
        for chunk in semantic_chunks:
            text = chunk.get("text", "")
            if text not in unique_chunks:
                unique_chunks[text] = chunk
        
        # Then add keyword matches
        for chunk in keyword_matches:
            text = chunk.get("text", "")
            if text not in unique_chunks:
                unique_chunks[text] = chunk
    
    # Return top-k
    result_chunks = list(unique_chunks.values())[:k]

    # In rag_pipeline/retriever/hybrid_retrieve.py

    # Add after line 175 (after result_chunks is created, before debug output):

    # Filter out tiny chunks (section headers, etc.) that don't provide useful content
    # This aligns with confidence_gate.py's MIN_CONTEXT_CHARS requirement
    MIN_CHUNK_CHARS = 200
    filtered_chunks = [c for c in result_chunks if len(c.get('text', '')) >= MIN_CHUNK_CHARS]
    
    # Only use filtered chunks if we still have enough (at least 2 for confidence gate)
    # Otherwise, keep original chunks to avoid over-filtering
    if len(filtered_chunks) >= 2:
        result_chunks = filtered_chunks[:k]  # Re-apply top-k limit
    # else: keep original result_chunks (don't filter if it would leave too few)

    print(f"DEBUG: Final result_chunks count={len(result_chunks)}")
    total_chars = sum(len(c.get('text', '')) for c in result_chunks)
    print(f"DEBUG: Total context chars={total_chars}")
    for i, c in enumerate(result_chunks):
        print(f"  Final chunk {i+1}: {c['metadata'].get('section', 'unknown')} ({len(c.get('text', ''))} chars)")
    
    return result_chunks