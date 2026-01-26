# rag_pipeline/verifier/confidence_gate.py

from typing import List, Dict, Tuple, Optional
import re

# Thresholds (tune empirically)
MIN_CHUNKS_REQUIRED = 1  # At least 1 substantial chunk is enough
MIN_CONTEXT_CHARS = 200


def check_intent_evidence(chunks: List[Dict], intent_types: Optional[List[str]] = None) -> Tuple[bool, int]:
    """
    Check if chunks contain intent evidence (pattern hits or high intent match count).
    
    Args:
        chunks: List of chunk dicts
        intent_types: Optional list of intent types (for pattern matching)
    
    Returns:
        (has_evidence, evidence_count)
    """
    if not intent_types:
        # If no intent specified, just check if chunks are substantial
        substantial = [c for c in chunks if len(c.get("text", "")) >= MIN_CONTEXT_CHARS]
        return len(substantial) > 0, len(substantial)
    
    # Import here to avoid circular dependency
    from rag_pipeline.retriever.intent_classifier import get_evidence_patterns
    
    evidence_patterns = get_evidence_patterns(intent_types)
    evidence_count = 0
    
    for chunk in chunks:
        text = chunk.get("text", "")
        for pattern in evidence_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                evidence_count += 1
                break  # Count once per chunk
    
    has_evidence = evidence_count > 0
    return has_evidence, evidence_count


def compute_confidence(chunks: List[Dict], verification_passed: bool, 
                      intent_types: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Returns (should_answer, reason_if_abstain)
    
    Generalized confidence gate that works across intent types.
    Allows MIN_CHUNKS_REQUIRED=1 if at least one substantial chunk exists and has intent evidence.
    """
    if len(chunks) < MIN_CHUNKS_REQUIRED:
        return False, f"Insufficient context: only {len(chunks)} chunks retrieved."
    
    total_context = sum(len(c.get("text", "")) for c in chunks)
    if total_context < MIN_CONTEXT_CHARS:
        return False, f"Context too short ({total_context} chars)."
    
    # Check for intent evidence if intent types provided
    if intent_types:
        has_evidence, evidence_count = check_intent_evidence(chunks, intent_types)
        if not has_evidence and len(chunks) == 1:
            # Single chunk without evidence is suspicious
            return False, f"Single chunk lacks intent evidence (intent: {intent_types})."
    
    if not verification_passed:
        return False, "Answer failed grounding verification."
    
    return True, ""
