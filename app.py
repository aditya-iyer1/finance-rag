import os
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline.llm.generator import generate_answer_with_gate
from rag_pipeline.retriever.chroma_client import list_indexed_doc_ids
from rag_pipeline.retriever.retrieve import query_chunks


load_dotenv()
PDF_DIR = "data/raw-pdfs"

st.set_page_config(page_title="Finance Filing Q&A", layout="wide")


def render_result_box(body: str, abstained: bool) -> None:
    if abstained:
        st.warning(body)
    else:
        st.success(body)


def format_distance(distance) -> str:
    if distance is None:
        return "N/A"
    try:
        return f"{float(distance):.4f}"
    except (TypeError, ValueError):
        return str(distance)


def clean_section_label(section: str) -> str:
    return section.rsplit(" (part ", 1)[0] if section else "unknown"


def format_chunk_id(chunk_id) -> str:
    try:
        return f"chunk {int(chunk_id)}"
    except (TypeError, ValueError):
        return "chunk unknown"


def build_excerpt(text: str, max_chars: int = 110) -> str:
    excerpt = " ".join((text or "").split())
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[: max_chars - 1].rstrip() + "…"


def build_citation_map(chunks: List[Dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        section = clean_section_label(metadata.get("section", "unknown"))
        doc_id = metadata.get("doc_id", "unknown")
        chunk_id = format_chunk_id(metadata.get("chunk_id"))
        excerpt = build_excerpt(chunk.get("text", ""))
        lines.append(f"- [{i}] `{doc_id}` | `{section}` | {chunk_id}")
        lines.append(f'  "{excerpt}"')
    return "\n".join(lines)


def render_context_table(chunks: List[Dict]) -> None:
    rows = []
    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        rows.append(
            {
                "citation": f"[{i}]",
                "section": clean_section_label(metadata.get("section", "unknown")),
                "doc_id": metadata.get("doc_id", "unknown"),
                "chunk": format_chunk_id(metadata.get("chunk_id")),
                "distance": format_distance(chunk.get("distance")),
                "excerpt": build_excerpt(chunk.get("text", ""), max_chars=90),
            }
        )
    st.table(rows)


def list_available_pdfs() -> List[Path]:
    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.exists():
        return []
    return sorted(path for path in pdf_dir.iterdir() if path.suffix.lower() == ".pdf")


st.title("Finance Filing Q&A")
st.caption("Ask a question about the currently indexed filing.")

with st.sidebar:
    st.subheader("Active Document")
    available_pdfs = list_available_pdfs()
    indexed_doc_ids = list_indexed_doc_ids()

    if indexed_doc_ids:
        active_doc_id = st.selectbox("Query this filing", indexed_doc_ids, index=0)
        st.caption(f"Active document: `{active_doc_id}`")
    else:
        active_doc_id = None
        st.write("Current selection: none")

    st.caption(f"Available PDFs in `{PDF_DIR}`")
    if available_pdfs:
        for pdf_path in available_pdfs:
            doc_id = pdf_path.stem
            if doc_id == active_doc_id:
                marker = " (active)"
            elif doc_id in indexed_doc_ids:
                marker = " (indexed)"
            else:
                marker = ""
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            st.write(f"- `{pdf_path.name}` ({size_mb:.1f} MB){marker}")
    else:
        st.write("No PDF files found.")

    st.subheader("Retrieved Context")
    st.caption("Shown after a query runs.")

query = st.text_input("Question", placeholder="Where is the company headquartered?")

if st.button("Ask", type="primary", use_container_width=False):
    cleaned_query = query.strip()

    if not cleaned_query:
        st.info("Enter a question to query the indexed document.")
    elif not active_doc_id:
        st.info("Index a document first, then select it as the active document.")
    else:
        try:
            with st.spinner("Retrieving context..."):
                chunks = query_chunks(cleaned_query, top_k=5, active_doc_id=active_doc_id)

            with st.spinner("Generating answer..."):
                answer, abstained = generate_answer_with_gate(cleaned_query, chunks)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
        else:
            st.subheader("Answer")
            st.caption(f"Active document: `{active_doc_id}`")
            render_result_box(answer, abstained)

            if chunks:
                st.markdown("**Citations**")
                st.markdown(build_citation_map(chunks))
            else:
                st.caption("No chunks were retrieved for this query.")

            with st.sidebar:
                with st.expander("Retrieved Context", expanded=False):
                    if chunks:
                        render_context_table(chunks)
                    else:
                        st.write("No retrieved chunks.")


if not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.warning("`OPENAI_API_KEY` is not set.")
