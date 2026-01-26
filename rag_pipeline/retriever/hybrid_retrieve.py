# rag_pipeline/retriever/hybrid_retrieve.py

from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions

import os

# Initialize Chroma vector DB
def load_vector_db(persist_path="data/chroma_index"):
    return Chroma(
        persist_directory=persist_path,
        embedding_function=OpenAIEmbeddings()
    )

def vector_retrieve(query: str, db, k: int = 5) -> List[Document]:
    return db.similarity_search(query, k=k)

def vector_retrieve_with_score(query: str, db, k: int = 5):
    return db.similarity_search_with_score(query, k=k)

def keyword_filter(chunks: List[Document], query: str) -> List[Document]:
    query_lower = query.lower()
    return [
        doc for doc in chunks 
        if query_lower in doc.page_content.lower() or
           any(query_lower in str(v).lower() for v in doc.metadata.values())
    ]

def hybrid_retrieve(query: str, k: int = 5) -> List[Dict]:
    db = load_vector_db()
    # Get chunks with scores for distance information
    semantic_chunks_with_scores = vector_retrieve_with_score(query, db, k=k * 2)
    semantic_chunks = [doc for doc, _ in semantic_chunks_with_scores]
    chunk_scores = {doc.page_content: score for doc, score in semantic_chunks_with_scores}
    
    keyword_matches = keyword_filter(semantic_chunks, query)

    # Combine with deduplication
    unique_chunks = {doc.page_content: doc for doc in (semantic_chunks + keyword_matches)}
    
    # Convert to consistent dict format with metadata
    result_chunks = []
    for doc in list(unique_chunks.values())[:k]:
        # Ensure metadata is a dict and has required fields
        metadata = doc.metadata if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else {}
        
        # Ensure required metadata fields exist with defaults
        if "doc_id" not in metadata:
            metadata["doc_id"] = "unknown"
        if "chunk_id" not in metadata:
            metadata["chunk_id"] = "unknown"
        if "section" not in metadata:
            metadata["section"] = "unknown"
        
        result_chunks.append({
            "text": doc.page_content,
            "metadata": metadata,
            "distance": chunk_scores.get(doc.page_content, None)
        })
    
    return result_chunks