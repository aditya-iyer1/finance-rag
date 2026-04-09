import json
import logging
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def _build_context(chunks: List[Dict]) -> str:
    context_lines = []
    for i, c in enumerate(chunks):
        ref = f"[{i+1}]"
        section = c['metadata'].get('section', 'unknown')
        context_lines.append(f"{ref} [Section: {section}]\n{c['text']}")
    return "\n\n".join(context_lines)


def format_prompt(query: str, chunks: List[Dict]) -> str:
    context = _build_context(chunks)
    
    return f"""You are a financial analysis assistant.
Answer the question using ONLY the numbered context below.
Cite sources using bracketed numbers like [1], [2].

If the answer is partially present, respond with the relevant information and state where the context is incomplete.

If absolutely no relevant information exists, respond exactly with: "The provided document does not contain that information."

Context:
{context}

Question:
{query}

Answer (with citations):"""

def generate_answer(query: str, chunks: List[Dict], model_name: str = "gpt-4o", dry_run: bool = False) -> str:
    prompt = format_prompt(query, chunks)

    if dry_run:
        logger.info("DRY RUN - Prompt Only:\n%s", prompt)
        return prompt
        
        
    # Make sure this import is from langchain_openai.chat_models, not langchain.chat_models
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    messages = [
        SystemMessage(
            content="You are a financial analyst trained to answer questions based ONLY on provided SEC filings."
        ),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content
    return answer

def log_used_chunks(query: str, chunks: List[Dict], answer: str):
    logger.info("Query: %s", query)
    for i, c in enumerate(chunks):
        metadata = c.get('metadata', {})
        doc_id = metadata.get('doc_id', 'unknown')
        section = metadata.get('section', 'unknown')
        chunk_id = metadata.get('chunk_id', 'unknown')
        distance = c.get('distance')
        distance_str = f" (distance={distance:.4f})" if distance is not None else ""
        logger.info("  [%d] %s::%s::chunk-%s%s", i+1, doc_id, section, chunk_id, distance_str)
    logger.info("Answer length: %d chars", len(answer))

ABSTAIN_MESSAGE = "I cannot confidently answer this question based on the available document context."

def _format_structured_prompt(query: str, chunks: List[Dict]) -> str:
    context = _build_context(chunks)
    
    return f"""You are a financial analysis assistant.
Answer the question using ONLY the numbered context below.
Cite sources using bracketed numbers like [1], [2].

If the answer is partially present, respond with the relevant information and state where the context is incomplete.

You MUST respond with a JSON object containing exactly two fields:
- "is_answerable": boolean — true if the context contains relevant information to answer the question, false otherwise
- "answer": string — your answer with citations if answerable, or a brief explanation of why the context is insufficient

Context:
{context}

Question:
{query}"""


def generate_answer_with_gate(query: str, chunks: List[Dict], model_name: str = "gpt-4o") -> Tuple[str, bool]:
    """
    Returns (answer, did_abstain)
    """
    from rag_pipeline.verifier.confidence_gate import compute_confidence
    from rag_pipeline.verifier.hallucination_guard import sentence_overlap
    from rag_pipeline.retriever.intent_classifier import classify_intent
    
    # Pre-generation gate: check retrieval quality
    if len(chunks) == 0:
        return ABSTAIN_MESSAGE + " (no relevant chunks found)", True
    
    # Classify intent for confidence gate
    detected_intents, _ = classify_intent(query)
    
    # Use structured JSON output for reliable abstention detection
    prompt = _format_structured_prompt(query, chunks)
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    messages = [
        SystemMessage(
            content="You are a financial analyst trained to answer questions based ONLY on provided SEC filings. Always respond with valid JSON."
        ),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse structured response
    try:
        result = json.loads(response.content)
        answer = result.get("answer", response.content)
        is_answerable = result.get("is_answerable", True)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse structured JSON from LLM response, falling back to raw content")
        answer = response.content
        is_answerable = True
    
    # LLM explicitly indicated it cannot answer
    if not is_answerable:
        return ABSTAIN_MESSAGE, True
    
    # Post-generation gate: verification
    verification_passed = sentence_overlap(answer, chunks)
    should_answer, reason = compute_confidence(chunks, verification_passed, intent_types=detected_intents if detected_intents else None)
    
    if not should_answer:
        return f"{ABSTAIN_MESSAGE} Reason: {reason}", True
    
    log_used_chunks(query, chunks, answer)
    return answer, False