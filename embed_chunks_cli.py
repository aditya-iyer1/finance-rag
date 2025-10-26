# embed_chunks_cli.py

from rag_pipeline.parser.pdf_loader import parse_pdf_sections
from rag_pipeline.parser.chunker import chunk_all_sections
from rag_pipeline.retriever.embed_store import embed_chunks, store_in_chroma

def main():
    print("🔍 Parsing PDF...")
    sections = parse_pdf_sections("data/raw-pdfs/tesla-2024-10K.pdf")
    
    print(f"📄 Parsed {len(sections)} sections")
    chunks = chunk_all_sections(sections)
    print(f"🔗 Chunked into {len(chunks)} chunks")

    print("🧠 Embedding...")
    embedded_chunks = embed_chunks(chunks)


        # Attach missing metadata and unique IDs to each chunk
    for i, chunk in enumerate(embedded_chunks):
        if "metadata" not in chunk:
            chunk["metadata"] = {"section": chunk.get("section", "unknown")}
        if "id" not in chunk:
            chunk["id"] = f"chunk-{i}"
        
    print("📦 Storing in ChromaDB...")
    store_in_chroma(embedded_chunks)

    print("✅ Done!")

if __name__ == "__main__":
    main()