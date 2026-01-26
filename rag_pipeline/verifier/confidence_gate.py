# rag_pipeline/verifier/confidence_gate.py

from typing import List, Dict, Tuple

# Thresholds (tune empirically)
MIN_CHUNKS_REQUIRED = 2
MIN_CONTEXT_CHARS = 200

def compute_confidence(chunks: List[Dict], verification_passed: bool) -> Tuple[bool, str]:
    """
    Returns (should_answer, reason_if_abstain)
    
    Note: Distance-based gating removed as L2 distances vary widely and are not reliable
    indicators of relevance. We rely on chunk count, context size, and verification instead.
    """
    if len(chunks) < MIN_CHUNKS_REQUIRED:
        return False, f"Insufficient context: only {len(chunks)} chunks retrieved."
    
    total_context = sum(len(c.get("text", "")) for c in chunks)
    if total_context < MIN_CONTEXT_CHARS:
        return False, f"Context too short ({total_context} chars)."
    
    # Removed distance check - L2 distances are not reliable indicators
    # If needed, use percentile-based approach or similarity score normalization
    
    if not verification_passed:
        return False, "Answer failed grounding verification."
    
    return True, ""
