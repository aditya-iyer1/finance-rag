# embed_chunks_cli.py

import argparse
import logging
import os
from rag_pipeline.parser.pdf_loader import parse_pdf_sections
from rag_pipeline.parser.chunker import chunk_all_sections
from rag_pipeline.retriever.embed_store import embed_chunks, store_in_chroma
from rag_pipeline.retriever.chroma_client import (
    DEFAULT_PERSIST_DIR,
    delete_doc_id,
    list_indexed_doc_ids,
)

PDF_DIR = "data/raw-pdfs"


def find_pdfs():
    """Scan PDF_DIR for .pdf files and return sorted list of paths."""
    if not os.path.isdir(PDF_DIR):
        return []
    return [
        os.path.join(PDF_DIR, f)
        for f in sorted(os.listdir(PDF_DIR))
        if f.lower().endswith(".pdf")
    ]

def get_indexed_doc_ids():
    """Backwards-compatible wrapper returning indexed doc_ids as a set."""
    return set(list_indexed_doc_ids(DEFAULT_PERSIST_DIR))


def select_pdf(pdfs):
    """Display numbered PDF list and return the user's choice."""
    print("\nAvailable PDFs in data/raw-pdfs/:\n")
    for i, path in enumerate(pdfs, 1):
        name = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{i}] {name}  ({size_mb:.1f} MB)")

    print()
    while True:
        choice = input(f"Select a PDF to index (1-{len(pdfs)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(pdfs):
            return pdfs[int(choice) - 1]
        print(f"  Invalid choice. Enter a number between 1 and {len(pdfs)}.")


def main():
    parser = argparse.ArgumentParser(description="Parse, chunk, embed, and store a PDF in ChromaDB.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show section list, metadata samples, and debug info")
    args = parser.parse_args()
    verbose = args.verbose

    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s - %(levelname)s - %(message)s")

    # --- PDF selection ---
    pdfs = find_pdfs()
    if not pdfs:
        print(f"No PDF files found in {PDF_DIR}/")
        print("Place your 10-K filing PDFs there and re-run.")
        return

    pdf_path = select_pdf(pdfs)
    pdf_name = os.path.basename(pdf_path)
    doc_id = os.path.splitext(pdf_name)[0]

    # --- Check existing index ---
    existing_ids = get_indexed_doc_ids()
    if existing_ids:
        if doc_id in existing_ids and len(existing_ids) == 1:
            print(f"\n  Index already contains {doc_id}.")
            answer = input("  Re-index this document? This will replace only this document's stored chunks. (y/N): ").strip().lower()
            if answer != "y":
                print("  Skipped.")
                return
            deleted = delete_doc_id(doc_id, DEFAULT_PERSIST_DIR)
            print(f"  Removed {deleted} existing chunks for {doc_id}")
        elif doc_id in existing_ids:
            print(f"\n  Index already contains {doc_id} alongside: {', '.join(sorted(existing_ids - {doc_id}))}")
            answer = input("  Re-index this document? This will replace only this document's stored chunks. (y/N): ").strip().lower()
            if answer != "y":
                print("  Skipped.")
                return
            deleted = delete_doc_id(doc_id, DEFAULT_PERSIST_DIR)
            print(f"  Removed {deleted} existing chunks for {doc_id}")
        else:
            other_docs = ", ".join(sorted(existing_ids))
            print(f"\n  Index currently contains: {other_docs}")
            print(f"  Indexing {doc_id} will add it to the shared index.")
            print("  Queries remain scoped to one active document at a time.")

    # --- Parse ---
    print(f"\nParsing {pdf_name}...", end=" ", flush=True)
    sections = parse_pdf_sections(pdf_path)
    print(f"{len(sections)} sections found")

    if verbose:
        for i, section_title in enumerate(sections.keys()):
            print(f"  {i+1}. {section_title}")

    # --- Chunk ---
    chunk_size = 512
    overlap = 50
    print("Chunking...", end=" ", flush=True)
    chunks = chunk_all_sections(sections, chunk_size=chunk_size, overlap=overlap)
    print(f"{len(chunks)} chunks ({chunk_size} tokens, {overlap} overlap)")

    # --- Embed ---
    print("Embedding...", end=" ", flush=True)
    embedded_chunks = embed_chunks(chunks, show_progress=verbose)
    print("done")

    # --- Attach metadata ---
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

    # --- Store ---
    print("Storing in ChromaDB...", end=" ", flush=True)
    store_in_chroma(embedded_chunks, verbose=verbose)
    print(f"{len(embedded_chunks)} chunks indexed")

    final_ids = sorted(get_indexed_doc_ids())
    print(f"\nDone. Indexed document: {doc_id}")
    print(f"Available indexed documents: {', '.join(final_ids)}")

    if verbose:
        print("\nTo add or refresh another document, re-run this script and select a PDF.")
        print("Queries should always specify the active document they target.")


if __name__ == "__main__":
    main()
