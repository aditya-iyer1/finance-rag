# embed_chunks_cli.py

import os
import shutil
from rag_pipeline.parser.pdf_loader import parse_pdf_sections
from rag_pipeline.parser.chunker import chunk_all_sections
from rag_pipeline.retriever.embed_store import embed_chunks, store_in_chroma

def main():
    # Re-indexing instructions:
    # If you see "unknown" metadata in logs, the existing ChromaDB index was created
    # before metadata fields (doc_id, chunk_id) were added. To fix:
    # 1. Delete or rename the existing data/chroma_index directory
    # 2. Re-run this script to create a fresh index with full metadata
    #
    # Uncomment the lines below to automatically clear old index:
    # persist_dir = "data/chroma_index"
    # if os.path.exists(persist_dir):
    #     print(f"⚠️  Clearing old index at {persist_dir}...")
    #     shutil.rmtree(persist_dir)
    #     print("✅ Old index cleared. Creating new index with metadata...")
    print("🔍 Parsing PDF...")
    pdf_path = "data/raw-pdfs/tesla-2024-10K.pdf"
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]  # "tesla-2024-10K"
    sections = parse_pdf_sections(pdf_path)
    
    print(f"📄 Parsed {len(sections)} sections")

    for i, section_title in enumerate(sections.keys()):
        print(f"{i+1}. {section_title}")

    
    chunks = chunk_all_sections(sections)
    print(f"🔗 Chunked into {len(chunks)} chunks")

    print("🧠 Embedding...")
    embedded_chunks = embed_chunks(chunks)


    # Attach metadata and unique IDs to each chunk
    print("📝 Adding metadata (doc_id, section, chunk_id)...")
    for i, chunk in enumerate(embedded_chunks):
        chunk["metadata"] = {
            "doc_id": doc_id,
            "section": chunk.get("section", "unknown"),
            "chunk_id": i
        }
        chunk["id"] = f"{doc_id}::chunk-{i}"
    
    # Verify metadata before storing
    sample_metadata = embedded_chunks[0].get("metadata", {})
    print(f"✅ Sample metadata: {sample_metadata}")
        
    print("📦 Storing in ChromaDB...")
    store_in_chroma(embedded_chunks)

    print("✅ Done! Metadata includes doc_id, section, and chunk_id.")
    print("\n📋 Re-indexing Instructions:")
    print("   If you see 'unknown' metadata in logs, the existing index lacks metadata fields.")
    print("   To fix: Delete data/chroma_index/ and re-run this script.")
    print("   Or uncomment the auto-clear code at the top of main() to clear automatically.")

if __name__ == "__main__":
    main()