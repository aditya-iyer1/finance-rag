# rag_pipeline/verifier/hallucination_guard.py

import re
from difflib import SequenceMatcher
from typing import List, Dict

def clean_text(text: str) -> str:
    # Normalize whitespace and lowercase for simple matching
    return re.sub(r'\s+', ' ', text.strip().lower())

def sentence_overlap(answer, chunks):
    context_text = " ".join(c['text'] for c in chunks)
    
    # If the model explicitly says it's not in the document, assume it's correct
    if answer.strip() == "The provided document does not contain that information.":
        return True

    # Check if any sentence (or part of it) from the answer appears in the context
    return any(sentence.strip().lower() in context_text.lower() for sentence in answer.split('.'))