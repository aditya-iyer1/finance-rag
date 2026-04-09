# rag_pipeline/verifier/hallucination_guard.py

import json
import logging
import re
from typing import List, Dict

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("rag_pipeline")

_JUDGE_TEMPLATE = """You are a factual verification judge. Determine whether the ANSWER is fully supported by the CONTEXT.

Rules:
- "We", "our", "the Company" in the CONTEXT refer to the company being discussed.
- The ANSWER must not contain any facts, names, locations, or numbers that contradict the CONTEXT.
- If the ANSWER introduces specific claims (locations, figures, dates) not present in or contradicting the CONTEXT, it is NOT grounded.
- Minor paraphrasing is acceptable as long as the facts are preserved.

CONTEXT:
{CONTEXT_PLACEHOLDER}

ANSWER:
{ANSWER_PLACEHOLDER}

Respond with a JSON object: {"is_grounded": true/false, "reason": "brief explanation"}"""


def sentence_overlap(answer: str, chunks: List[Dict]) -> bool:
    """Check if the answer is factually grounded in the provided chunks using LLM-as-a-Judge."""
    stripped_answer = re.sub(r'\[\d+\]', '', answer).strip()
    if not stripped_answer:
        return False

    context_text = "\n\n".join(c['text'] for c in chunks)

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        prompt = _JUDGE_TEMPLATE.replace("{CONTEXT_PLACEHOLDER}", context_text).replace("{ANSWER_PLACEHOLDER}", stripped_answer)

        messages = [
            SystemMessage(content="You are a factual verification judge. Always respond with valid JSON."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        result = json.loads(response.content)
        return result.get("is_grounded", False)
    except Exception as e:
        logger.warning(f"LLM judge verification failed, falling back to word overlap: {e}")
        return _word_overlap_fallback(stripped_answer, chunks)


def _word_overlap_fallback(answer: str, chunks: List[Dict]) -> bool:
    """Fallback word-overlap check if LLM judge is unavailable."""
    context_words = set(re.findall(r'[a-z0-9]+', " ".join(c['text'] for c in chunks).lower()))

    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    if not sentences:
        return False

    for sentence in sentences:
        sentence_words = set(re.findall(r'[a-z0-9]+', sentence.lower()))
        if not sentence_words:
            continue
        overlap = sentence_words & context_words
        if len(overlap) / len(sentence_words) >= 0.5:
            return True

    return False
