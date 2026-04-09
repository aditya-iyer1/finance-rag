# rag_pipeline/retriever/hybrid_retrieve.py

"""
Generalized hybrid retrieval with intent-based scoring and fallback.
Works across many finance question types without brittle query-specific fixes.
"""

import logging
import re
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer
from rag_pipeline.retriever.chroma_client import get_collection, is_single_doc_mode
from rag_pipeline.retriever.retrieve import get_embedder
from rag_pipeline.retriever.intent_classifier import (
    classify_intent, get_intent_synonyms, get_evidence_patterns, extract_entity_terms
)

logger = logging.getLogger(__name__)

# Embedding model must match ingestion (same as retrieve.py)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Constants
MIN_CHUNK_CHARS = 200


def score_chunk(chunk: Dict, intent_types: List[str], intent_synonyms: set, 
                evidence_patterns: List[str], entity_terms: List[str], 
                single_doc_mode: bool) -> Tuple[float, int, int, int, bool]:
    """
    Score a chunk based on intent coverage and evidence patterns.
    
    Returns:
        (total_score, intent_match_count, evidence_pattern_hits, entity_match_count, has_evidence)
    """
    text_lower = chunk.get("text", "").lower()
    text_original = chunk.get("text", "")  # Keep original for capitalization
    metadata = chunk.get("metadata", {})
    metadata_str = " ".join(str(v).lower() for v in metadata.values())
    combined_text = f"{text_lower} {metadata_str}"
    
    # Count intent matches (using word boundaries)
    intent_match_count = 0
    for synonym in intent_synonyms:
        pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
        if re.search(pattern, combined_text, re.IGNORECASE):
            intent_match_count += 1
    
    # Count evidence pattern hits
    evidence_pattern_hits = 0
    for pattern in evidence_patterns:
        if re.search(pattern, text_original, re.IGNORECASE):
            evidence_pattern_hits += 1
    
    # Count entity matches (downweighted in single-doc mode)
    entity_match_count = 0
    if entity_terms:
        for entity_term in entity_terms:
            pattern = r'\b' + re.escape(entity_term.lower()) + r'\b'
            if re.search(pattern, combined_text, re.IGNORECASE):
                entity_match_count += 1
    
    # Scoring formula:
    # score = 10*intent_match_count + 3*evidence_pattern_hits + entity_weight*entity_match_count
    # Entity weight: 0.1 in single-doc mode, 0.5 in multi-doc mode
    entity_weight = 0.1 if single_doc_mode else 0.5
    
    intent_score = 10 * intent_match_count
    evidence_score = 3 * evidence_pattern_hits
    entity_score = entity_weight * entity_match_count
    
    # Small bonus for longer chunks (up to cap of 5)
    chunk_length = len(chunk.get("text", ""))
    length_bonus = min(chunk_length / 1000, 5.0)  # Cap at 5
    
    total_score = intent_score + evidence_score + entity_score + length_bonus
    has_evidence = evidence_pattern_hits > 0
    
    return total_score, intent_match_count, evidence_pattern_hits, entity_match_count, has_evidence


def keyword_filter(chunks: List[Dict], intent_types: List[str], intent_synonyms: set,
                   evidence_patterns: List[str], entity_terms: List[str], 
                   single_doc_mode: bool) -> List[Dict]:
    """
    Filter and score chunks based on intent coverage and evidence patterns.
    Returns chunks sorted by score (descending).
    """
    scored_chunks = []
    
    for chunk in chunks:
        score, intent_matches, evidence_hits, entity_matches, has_evidence = score_chunk(
            chunk, intent_types, intent_synonyms, evidence_patterns, 
            entity_terms, single_doc_mode
        )
        
        # Only include chunks with at least one match
        if intent_matches > 0 or evidence_hits > 0 or entity_matches > 0:
            chunk_length = len(chunk.get("text", ""))
            scored_chunks.append((
                chunk, score, intent_matches, evidence_hits, 
                entity_matches, has_evidence, chunk_length
            ))
    
    # Sort by: score desc, intent_matches desc, evidence_hits desc, length desc
    scored_chunks.sort(key=lambda x: (x[1], x[2], x[3], x[6]), reverse=True)
    
    if scored_chunks:
        logger.debug("Top keyword matches (score, intent, evidence, entity, has_evidence, length):")
        for i, (chunk, score, im, eh, em, he, length) in enumerate(scored_chunks[:3]):
            section = chunk.get('metadata', {}).get('section', 'unknown')
            logger.debug("  [%d] %s: score=%.2f (intent=%d, evidence=%d, entity=%d, has_evidence=%s, len=%d)",
                         i+1, section, score, im, eh, em, he, length)
    
    # Return just the chunks
    return [chunk for chunk, _, _, _, _, _, _ in scored_chunks]


def check_semantic_intent_coverage(semantic_chunks: List[Dict], intent_types: List[str],
                                   intent_synonyms: set, evidence_patterns: List[str]) -> Tuple[bool, int, int]:
    """
    Check if semantic top-k has good intent coverage.
    
    Returns:
        (has_good_coverage, intent_match_count, evidence_pattern_hits)
    """
    intent_match_count = 0
    evidence_pattern_hits = 0
    
    for chunk in semantic_chunks:
        text_lower = chunk.get("text", "").lower()
        text_original = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        metadata_str = " ".join(str(v).lower() for v in metadata.values())
        combined_text = f"{text_lower} {metadata_str}"
        
        # Count intent matches
        for synonym in intent_synonyms:
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, combined_text, re.IGNORECASE):
                intent_match_count += 1
                break  # Count once per chunk
        
        # Count evidence pattern hits
        for pattern in evidence_patterns:
            if re.search(pattern, text_original, re.IGNORECASE):
                evidence_pattern_hits += 1
                break  # Count once per chunk
    
    # Good coverage if: intent matches > 0 OR evidence hits > 0
    has_good_coverage = intent_match_count > 0 or evidence_pattern_hits > 0
    
    return has_good_coverage, intent_match_count, evidence_pattern_hits


def hybrid_retrieve(query: str, k: int = 5, persist_dir: str = "data/chroma_index", debug: bool = False) -> List[Dict]:
    """
    Generalized hybrid retrieval with intent-based scoring.
    
    Args:
        query: User query
        k: Number of chunks to return
        persist_dir: ChromaDB persist directory
        debug: Enable debug-level logging for this call
    
    Returns:
        List of chunk dicts with text, metadata, distance
    """
    parent_logger = logging.getLogger("rag_pipeline")
    prev_parent_level = parent_logger.level
    prev_level = logger.level
    if debug:
        parent_logger.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    try:
        return _hybrid_retrieve_impl(query, k, persist_dir, debug)
    finally:
        if debug:
            parent_logger.setLevel(prev_parent_level)
            logger.setLevel(prev_level)
    
def _hybrid_retrieve_impl(query: str, k: int, persist_dir: str, debug: bool) -> List[Dict]:
    collection = get_collection(persist_dir)
    
    # Detect single-doc mode (downweight entity terms)
    single_doc = is_single_doc_mode(persist_dir)
    logger.debug("Single-doc mode: %s", single_doc)
    
    # Classify query intent
    detected_intents, intent_confidence = classify_intent(query)
    intent_synonyms = get_intent_synonyms(detected_intents) if detected_intents else set()
    evidence_patterns = get_evidence_patterns(detected_intents) if detected_intents else []
    entity_terms = extract_entity_terms(query)
    
    logger.debug("Detected intents: %s, confidence: %.2f", detected_intents, intent_confidence)
    logger.debug("Entity terms: %s", entity_terms)
    logger.debug("Intent synonyms: %s...", list(intent_synonyms)[:5])
    
    # Semantic retrieval first
    embedder = get_embedder()
    query_embedding = embedder.encode([query])[0].tolist()
    
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
    
    # Check semantic intent coverage
    semantic_has_coverage, semantic_intent_matches, semantic_evidence_hits = check_semantic_intent_coverage(
        semantic_chunks[:k], detected_intents, intent_synonyms, evidence_patterns
    )
    
    # Decide on fallback: use keyword search if intent confidence is high BUT semantic has poor coverage
    use_keyword_fallback = False
    if detected_intents and intent_confidence > 0.3:  # Intent detected with reasonable confidence
        if not semantic_has_coverage or semantic_intent_matches == 0:
            use_keyword_fallback = True
        elif semantic_evidence_hits == 0 and len(evidence_patterns) > 0:
            # Semantic results contain generic synonym matches but no specific evidence patterns.
            # This means the results are tangentially related, not directly answering the query.
            use_keyword_fallback = True
    
    logger.debug("Semantic coverage: has_coverage=%s, intent_matches=%d, evidence_hits=%d",
                 semantic_has_coverage, semantic_intent_matches, semantic_evidence_hits)
    logger.debug("Use keyword fallback: %s", use_keyword_fallback)
    
    # Score semantic chunks
    scored_semantic = []
    for chunk in semantic_chunks:
        score, im, eh, em, he = score_chunk(
            chunk, detected_intents, intent_synonyms, evidence_patterns,
            entity_terms, single_doc
        )
        if im > 0 or eh > 0 or em > 0:
            scored_semantic.append((chunk, score, im, eh, em, he))
    
    # If fallback needed, do keyword search across all chunks
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
        
        # Filter and score all chunks
        keyword_matches = keyword_filter(
            all_chunks, detected_intents, intent_synonyms,
            evidence_patterns, entity_terms, single_doc
        )
        
        # Filter out tiny chunks
        keyword_matches = [c for c in keyword_matches if len(c.get('text', '')) >= MIN_CHUNK_CHARS]
        
        logger.debug("Keyword fallback found %d matches", len(keyword_matches))
        
        # Use keyword matches (they're scored and sorted)
        result_chunks = keyword_matches[:k]
    else:
        # Use semantic results (already scored)
        scored_semantic.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        result_chunks = [chunk for chunk, _, _, _, _, _ in scored_semantic[:k]]
        
        # Backfill with remaining semantic chunks if we have fewer than k scored results
        if len(result_chunks) < k:
            scored_ids = {id(chunk) for chunk, _, _, _, _, _ in scored_semantic}
            for chunk in semantic_chunks:
                if len(result_chunks) >= k:
                    break
                if id(chunk) not in scored_ids:
                    result_chunks.append(chunk)
    
    # Final filtering: remove tiny chunks if we have substantial ones
    filtered_chunks = [c for c in result_chunks if len(c.get('text', '')) >= MIN_CHUNK_CHARS]
    
    if len(filtered_chunks) >= 1:
        result_chunks = filtered_chunks[:k]
    else:
        # Fallback: keep largest chunks if filtering removed everything
        result_chunks = sorted(result_chunks, key=lambda c: len(c.get('text', '')), reverse=True)[:k]
    
    logger.debug("Final result_chunks count=%d", len(result_chunks))
    total_chars = sum(len(c.get('text', '')) for c in result_chunks)
    logger.debug("Total context chars=%d", total_chars)
    for i, c in enumerate(result_chunks):
        section = c.get('metadata', {}).get('section', 'unknown')
        length = len(c.get('text', ''))
        logger.debug("  Final chunk %d: %s (%d chars)", i+1, section, length)
    
    return result_chunks
