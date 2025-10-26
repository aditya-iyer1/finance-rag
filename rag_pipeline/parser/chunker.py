# rag_pipeline/parser/chunker.py

from typing import List, Dict
import tiktoken


def get_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def chunk_text(
    text: str,
    section: str,
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "gpt-3.5-turbo"
) -> List[Dict]:
    """Split long text into token-aware chunks with optional overlap."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append({
            "section": section,
            "chunk_id": chunk_id,
            "text": chunk_text,
            "tokens": len(chunk_tokens)
        })

        start += chunk_size - overlap
        chunk_id += 1

    return chunks


def chunk_all_sections(
    sections: Dict[str, str],
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "gpt-3.5-turbo"
) -> List[Dict]:
    """Apply chunking to all parsed sections."""
    all_chunks = []
    for section, text in sections.items():
        section_chunks = chunk_text(text, section, chunk_size, overlap, model)
        all_chunks.extend(section_chunks)
    return all_chunks