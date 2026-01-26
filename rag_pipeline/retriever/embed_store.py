from sentence_transformers import SentenceTransformer
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


def store_in_chroma(embedded_chunks, persist_dir="data/chroma_index"):
    """
    Store chunks in ChromaDB using the unified client.
    This ensures embed_chunks_cli.py writes to the same index that query_chunks() reads from.
    """
    print("📦 Storing in ChromaDB...")
    
    # Use unified client to get or create collection (for writing)
    from rag_pipeline.retriever.chroma_client import get_client, COLLECTION_NAME
    client = get_client(persist_dir)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

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