# embed_chunks_cli.py

import argparse
import logging
import os
import shutil
from rag_pipeline.parser.pdf_loader import parse_pdf_sections
from rag_pipeline.parser.chunker import chunk_all_sections
from rag_pipeline.retriever.embed_store import embed_chunks, store_in_chroma


def main():
    parser = argparse.ArgumentParser(description="Parse, chunk, embed, and store a PDF in ChromaDB.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show section list, metadata samples, and debug info")
    args = parser.parse_args()
    verbose = args.verbose

    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s - %(levelname)s - %(message)s")

    # Re-indexing instructions:
    # If you see "unknown" metadata in logs, the existing ChromaDB index was created
    # before metadata fields (doc_id, chunk_id) were added. To fix:
    # 1. Delete or rename the existing data/chroma_index directory
    # 2. Re-run this script to create a fresh index with full metadata
    #
    # Uncomment the lines below to automatically clear old index:
    # persist_dir = "data/chroma_index"
    # if os.path.exists(persist_dir):
    #     print(f"Clearing old index at {persist_dir}...")
    #     shutil.rmtree(persist_dir)
    #     print("Old index cleared. Creating new index with metadata...")

    pdf_path = "data/raw-pdfs/tesla-2024-10K.pdf"
    pdf_name = os.path.basename(pdf_path)
    doc_id = os.path.splitext(pdf_name)[0]  # "tesla-2024-10K"

    print(f"Parsing {pdf_name}...", end=" ", flush=True)
    sections = parse_pdf_sections(pdf_path)
    print(f"{len(sections)} sections found")

    if verbose:
        for i, section_title in enumerate(sections.keys()):
            print(f"  {i+1}. {section_title}")

    chunk_size = 512
    overlap = 50
    print("Chunking...", end=" ", flush=True)
    chunks = chunk_all_sections(sections, chunk_size=chunk_size, overlap=overlap)
    print(f"{len(chunks)} chunks ({chunk_size} tokens, {overlap} overlap)")

    print("Embedding...", end=" ", flush=True)
    embedded_chunks = embed_chunks(chunks, show_progress=verbose)
    print("done")

    # Attach metadata and unique IDs to each chunk
    for i, chunk in enumerate(embedded_chunks):
        chunk["metadata"] = {
            "doc_id": doc_id,
            "section": chunk.get("section", "unknown"),
            "chunk_id": i
        }
        chunk["id"] = f"{doc_id}::chunk-{i}"

    if verbose:
        sample_metadata = embedded_chunks[0].get("metadata", {})
        print(f"  Sample metadata: {sample_metadata}")

    print("Storing in ChromaDB...", end=" ", flush=True)
    store_in_chroma(embedded_chunks, verbose=verbose)
    print(f"{len(embedded_chunks)} chunks indexed")

    if verbose:
        print("\nRe-indexing instructions:")
        print("  If you see 'unknown' metadata in logs, the existing index lacks metadata fields.")
        print("  To fix: delete data/chroma_index/ and re-run this script.")


if __name__ == "__main__":
    main()
