# rag_pipeline/retriever/intent_classifier.py

"""
Lightweight query intent classification for finance RAG.
Rule-based, no LLM required.
"""

from typing import List, Dict, Tuple, Set
import re


# Intent buckets with synonym lists
INTENT_SYNONYMS = {
    'HQ_LOCATION': {
        'headquartered', 'headquarters', 'headquarter',
        'based', 'base',
        'located', 'location', 'locate',
        'city', 'where', 'main office', 'principal office',
        'head office', 'corporate headquarters'
    },
    'INCORPORATION': {
        'incorporated', 'incorporation', 'formed',
        'state of incorporation', 'incorporated in',
        'domicile', 'domiciled'
    },
    'BUSINESS_OVERVIEW': {
        'what does', 'overview', 'business model', 'segments',
        'products', 'services', 'operations', 'operates',
        'describe', 'business', 'company does', 'does the company'
    },
    'FINANCIALS_REVENUE': {
        'revenue', 'revenues', 'total revenues', 'net sales',
        'sales', 'top line', 'year ended', 'fiscal year',
        'financial results', 'earnings', 'income statement',
        'revenue last year', 'annual revenue'
    },
    'RISKS': {
        'risk', 'risk factors', 'risks', 'uncertainties',
        'threats', 'challenges', 'adverse', 'material risks'
    },
    'AUDITOR': {
        'auditor', 'independent registered', 'accounting firm',
        'pricewaterhousecoopers', 'pwc', 'independent auditor',
        'registered public accounting firm', 'audit firm'
    }
}

# Evidence patterns per intent (regex patterns for scoring)
EVIDENCE_PATTERNS = {
    'HQ_LOCATION': [
        r'\b(headquartered|based|located)\s+(in|at)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?',  # "headquartered in Austin, Texas"
        r'\b(headquarters|head office)\s+(in|at|located)\s+[A-Z][a-z]+',
    ],
    'INCORPORATION': [
        r'\bincorporated\s+in\s+[A-Z][a-z]+',
        r'\bState\s+of\s+[A-Z][a-z]+',  # "State of Delaware"
        r'\bdomiciled\s+in\s+[A-Z][a-z]+',
    ],
    'FINANCIALS_REVENUE': [
        r'\b(Total\s+revenues?|Net\s+sales)\s+[^.]*(?:year\s+ended|fiscal\s+year)',
        r'\b\$[\d,]+(?:\.\d+)?\s+(?:million|billion|thousand)\s+[^.]*(?:revenue|sales)',
    ],
    'AUDITOR': [
        r'\bPricewaterhouseCoopers\s+LLP',
        r'\bindependent\s+registered\s+public\s+accounting\s+firm',
        r'\bPwC\b',
    ],
    'BUSINESS_OVERVIEW': [
        r'\b(design|develop|manufacture|sell|lease)\s+[^.]*',
        r'\b(products|services|segments)\s+[^.]*',
    ],
    'RISKS': [
        r'\b(risk|uncertainty|threat|challenge)\s+[^.]*',
        r'\bItem\s+1A[^.]*Risk\s+Factors',
    ],
}


def classify_intent(query: str) -> Tuple[List[str], float]:
    """
    Classify query intent and return detected intents with confidence.
    
    Args:
        query: User query string
        
    Returns:
        (detected_intents, confidence_score)
        - detected_intents: List of intent types (1-2 max, sorted by confidence)
        - confidence_score: Simple score based on matched synonyms (0.0-1.0)
    """
    query_lower = query.lower()
    intent_scores = {}
    
    # Score each intent based on synonym matches
    for intent_type, synonyms in INTENT_SYNONYMS.items():
        score = 0.0
        matches = 0
        
        for synonym in synonyms:
            # Use word boundaries for exact matching
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, query_lower, re.IGNORECASE):
                matches += 1
                # Longer synonyms get higher weight
                score += len(synonym.split()) * 0.5
        
        if matches > 0:
            intent_scores[intent_type] = score
    
    # Normalize confidence (simple: sum of scores / max possible)
    total_score = sum(intent_scores.values())
    max_possible = 10.0  # Rough upper bound
    confidence = min(total_score / max_possible, 1.0) if total_score > 0 else 0.0
    
    # Return top 1-2 intents (sorted by score, descending)
    sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
    detected_intents = [intent for intent, _ in sorted_intents[:2]]
    
    return detected_intents, confidence


def get_intent_synonyms(intent_types: List[str]) -> Set[str]:
    """Get all synonyms for given intent types."""
    synonyms = set()
    for intent_type in intent_types:
        if intent_type in INTENT_SYNONYMS:
            synonyms.update(INTENT_SYNONYMS[intent_type])
    return synonyms


def get_evidence_patterns(intent_types: List[str]) -> List[str]:
    """Get all evidence patterns for given intent types."""
    patterns = []
    for intent_type in intent_types:
        if intent_type in EVIDENCE_PATTERNS:
            patterns.extend(EVIDENCE_PATTERNS[intent_type])
    return patterns


def extract_entity_terms(query: str) -> List[str]:
    """
    Extract entity terms (proper nouns, company names) from query.
    Simple heuristic: capitalized words that aren't common words.
    """
    import re
    words = query.split()
    capitalized_words = []
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'where', 'who', 'when', 'how', 'does', 'do', 'did'}
    
    for word in words:
        # Remove punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word and clean_word[0].isupper() and clean_word.lower() not in common_words:
            capitalized_words.append(clean_word.lower())
    
    return list(set(capitalized_words))  # Deduplicate
