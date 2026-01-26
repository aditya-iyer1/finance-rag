from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict

def embed_chunks(
    chunks: List[Dict],
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
) -> List[Dict]:
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    return chunks


from chromadb import PersistentClient
import os

def store_in_chroma(embedded_chunks, persist_dir="data/chroma_index"):
    """
    Store chunks in ChromaDB using the same path resolution as retrieve.py.
    This ensures embed_chunks_cli.py writes to the same index that query_chunks() reads from.
    """
    print("📦 Storing in ChromaDB...")
    
    # Convert to absolute path relative to project root
    if not os.path.isabs(persist_dir):
        # Get project root (3 levels up from this file: rag_pipeline/retriever/embed_store.py -> project root)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        persist_dir = os.path.join(project_root, persist_dir)
    
    client = PersistentClient(path=persist_dir)

    # Collection name must match retrieve.py: "finance_rag"
    collection = client.get_or_create_collection(name="finance_rag")

    # Batch all chunks together for efficient storage
    documents = [chunk["text"] for chunk in embedded_chunks]
    embeddings = [chunk["embedding"] for chunk in embedded_chunks]
    metadatas = [chunk["metadata"] for chunk in embedded_chunks]
    ids = [chunk["id"] for chunk in embedded_chunks]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Stored {len(embedded_chunks)} chunks in ChromaDB (persistent storage at {persist_dir})")