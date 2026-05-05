# Hallucination-Resistant Finance RAG

A retrieval-augmented generation pipeline for answering questions over SEC filings with explicit grounding safeguards. The repository is built around a narrow but practical workflow:

1. Parse a filing PDF into sectioned text.
2. Chunk and embed the text into a persistent ChromaDB index.
3. Retrieve document-scoped evidence with semantic search plus intent-aware keyword fallback.
4. Generate an answer with citations.
5. Gate the answer with abstention and grounding checks before returning it.

The current repo is focused on 10-K style filings and includes a minimal Streamlit UI, an interactive indexing CLI, and validation scripts for the bundled Tesla and JPMorgan Chase sample filings.

## What This Project Does

- Indexes one or more filing PDFs from `data/raw-pdfs/`.
- Stores all indexed filings in one shared ChromaDB collection.
- Forces each query onto one active `doc_id` so retrieval does not mix companies.
- Uses a lightweight sentence-transformer embedding model for semantic retrieval.
- Adds rule-based intent classification and regex evidence checks to improve recall on common finance questions.
- Uses OpenAI models to generate the final answer and to verify that the answer is grounded in retrieved context.
- Abstains when retrieval or grounding is not strong enough.

## Architecture

```text
PDF -> section parser -> token-aware chunker -> embeddings -> ChromaDB
   -> active-document retrieval -> answer generation -> grounding gate -> final answer or abstain
```

### Ingestion

The indexing flow is driven by [`embed_chunks_cli.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/embed_chunks_cli.py).

- PDF parsing uses PyMuPDF in [`pdf_loader.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/parser/pdf_loader.py).
- The parser extracts raw text, splits it by `Item` headers, and normalizes section titles when possible.
- Chunking uses `tiktoken` in [`chunker.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/parser/chunker.py).
- Default chunking is 512 tokens with 50-token overlap.
- Embeddings are created with `sentence-transformers/paraphrase-MiniLM-L3-v2` in [`embed_store.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/retriever/embed_store.py).
- Chunks are stored in a persistent ChromaDB collection named `finance_rag`.

Each stored chunk carries metadata used throughout the rest of the pipeline:

- `doc_id`
- `section`
- `chunk_id`

That metadata is the basis for document scoping, traceability, and UI citations.

### Retrieval

Query entry starts in [`retrieve.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/retriever/retrieve.py) and resolves the active filing through [`chroma_client.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/retriever/chroma_client.py).

Important behavior:

- If no documents are indexed, querying fails immediately.
- If exactly one document is indexed, it becomes the implicit default.
- If multiple documents are indexed, the caller must provide `active_doc_id`.

Actual retrieval logic lives in [`hybrid_retrieve.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/retriever/hybrid_retrieve.py):

- The query is classified with a rule-based intent classifier from [`intent_classifier.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/retriever/intent_classifier.py).
- Semantic retrieval runs first against the active document only.
- Retrieved chunks are inspected for intent-synonym coverage and regex evidence patterns.
- If semantic results look weak for the detected intent, the pipeline falls back to a keyword-style scan over all chunks in the active document.
- Final chunk ranking prefers intent matches and evidence patterns, then lightly rewards chunk length.
- Very small chunks are filtered out when larger chunks are available.

The built-in intent buckets are:

- `HQ_LOCATION`
- `INCORPORATION`
- `BUSINESS_OVERVIEW`
- `FINANCIALS_REVENUE`
- `RISKS`
- `AUDITOR`

This is not a general natural-language router. It is a targeted rule set tuned for common filing questions.

### Generation

Answer generation lives in [`generator.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/llm/generator.py).

- The prompt includes numbered context blocks.
- The model is instructed to answer using only that context.
- Citations are expected in bracket form such as `[1]` and `[2]`.
- The gated path uses `gpt-4o` with JSON output requirements so the system can distinguish answerable from unanswerable cases programmatically.

Externally, the public helper returns:

- `answer: str`
- `did_abstain: bool`

### Verification and Abstention

The safeguard layer is split across two modules:

- [`confidence_gate.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/verifier/confidence_gate.py)
- [`hallucination_guard.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/rag_pipeline/verifier/hallucination_guard.py)

The checks are:

- No retrieved chunks means immediate abstention.
- The LLM can explicitly mark the question unanswerable.
- Retrieved context must clear minimum count and minimum character thresholds.
- When an intent is detected, a single retrieved chunk without intent evidence is treated as suspicious.
- The generated answer is checked for grounding with a second LLM pass using `gpt-4o-mini`.
- If the judge call fails, the verifier falls back to a simpler word-overlap heuristic.

If a check fails, the system returns an abstention message instead of a normal answer.

## Why The Pipeline Is Built This Way

### Hybrid retrieval instead of vector search only

Pure embedding similarity is often too loose for filing questions like headquarters, incorporation, auditor, or specific risk-factor lookups. This repo keeps semantic retrieval for breadth, then adds intent-aware rescoring and keyword fallback for precision. The tradeoff is more hand-built logic and less uniform ranking behavior.

### Active-document scoping instead of free multi-document search

All indexed filings share one collection, but every query is resolved to one active document. That is the main guard against cross-company contamination. The tradeoff is that callers have to specify `active_doc_id` once multiple filings are present.

### Structured abstention instead of free-form answer parsing

The generator asks the model for structured JSON internally, including whether the question is answerable. That makes abstention handling more reliable than trying to infer intent from arbitrary prose. The tradeoff is tighter dependence on the model following a structured output contract.

### LLM grounding check instead of lexical matching only

The post-generation verifier uses a second model pass because overlap-based checks alone are too weak for factual finance answers. The tradeoff is latency, cost, and dependence on another model call. A lightweight fallback still exists if the judge call fails.

## Repository Layout

```text
hallucination-resistant-finance-rag/
|-- app.py
|-- embed_chunks_cli.py
|-- run_validation.py
|-- run_validation_jpmc.py
|-- requirements.txt
|-- data/
|   `-- raw-pdfs/
|-- notebooks/
`-- rag_pipeline/
    |-- llm/
    |-- parser/
    |-- retriever/
    `-- verifier/
```

Key files:

- [`app.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/app.py): minimal Streamlit UI for querying indexed filings
- [`embed_chunks_cli.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/embed_chunks_cli.py): interactive indexing workflow
- [`run_validation.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/run_validation.py): Tesla-oriented validation matrix
- [`run_validation_jpmc.py`](/Users/aditya/Documents/projects/hallucination-resistant-finance-rag/run_validation_jpmc.py): JPMorgan Chase-oriented validation matrix

## Setup

### Requirements

- Python 3.10+ recommended
- An OpenAI API key in the environment

### Install

```bash
python -m venv rag
source rag/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
```

## Index A Filing

Run the interactive indexer:

```bash
python embed_chunks_cli.py
```

Verbose mode:

```bash
python embed_chunks_cli.py --verbose
```

What the CLI actually does:

1. Scans `data/raw-pdfs/` for PDFs.
2. Lets you choose a file from a numbered menu.
3. Parses the filing into sections.
4. Chunks the text.
5. Embeds all chunks.
6. Stores the chunks in ChromaDB.
7. If that `doc_id` already exists, offers to replace only that document's stored chunks.

Multiple documents can coexist in the same index. Re-indexing one filing does not wipe the others.

## Query The Pipeline

### Streamlit UI

```bash
streamlit run app.py
```

The UI:

- Lists indexed documents in the sidebar
- Lets you choose the active filing
- Accepts one question at a time
- Shows the answer or abstention
- Displays section-level citations and retrieved chunk excerpts

The UI does not perform indexing. You must run the CLI first.

### Python usage

```python
from rag_pipeline.retriever.retrieve import query_chunks
from rag_pipeline.llm.generator import generate_answer_with_gate

chunks = query_chunks(
    "Where is the company headquartered?",
    top_k=5,
    active_doc_id="tesla-2024-10K",
)

answer, abstained = generate_answer_with_gate(
    "Where is the company headquartered?",
    chunks,
)

print(answer)
print(abstained)
```

## Validation Scripts

The repository includes two validation runners:

```bash
python run_validation.py
python run_validation_jpmc.py
```

Optional flags:

```bash
python run_validation.py --debug
python run_validation.py --no-color
python run_validation.py --doc-id tesla-2024-10K
```

The validation scripts exercise three kinds of cases:

- Known-answer prompts that should produce grounded answers
- Out-of-scope prompts that should abstain
- Edge cases that should not crash the pipeline

The bundled validation matrices cover examples such as:

- Headquarters
- Incorporation
- Business overview
- Auditor
- Risk factors
- Deliberately unanswerable company-mismatch queries

## Example End-To-End Flow

1. Put a filing PDF in `data/raw-pdfs/`.
2. Run `python embed_chunks_cli.py` and select the file.
3. Query it through `streamlit run app.py` or from Python.
4. Inspect the returned citations and retrieved context.
5. If evidence is weak or unsupported, expect an abstention instead of a forced answer.

## Current Limitations

- The embedding model is lightweight and general-purpose, not finance-specific.
- Retrieval does not use a reranker.
- Intent classification is rule-based and limited to the built-in intent buckets.
- Section parsing depends on common 10-K `Item` formatting and may not generalize cleanly to every filing layout.
- Citations are section- and chunk-level, not page-level.
- The verifier uses extra LLM calls, which adds latency and cost.
- The Streamlit app is intentionally minimal and supports querying only, not indexing or richer document exploration.
- The pipeline is designed around one active document per query rather than true multi-document reasoning.

## Practical Extensions

Reasonable next steps for this codebase would be:

- Add a reranker after initial retrieval.
- Preserve PDF page numbers during parsing and carry them through metadata.
- Expand the intent classifier or replace it with a learned router.
- Add caching and cost instrumentation around generation and verification calls.
- Add stronger evaluation metrics than simple validation matrices.
- Broaden the UI beyond the current single-question workflow.

## Status

This repository already contains a working end-to-end pipeline:

- ingestion
- persistent vector storage
- active-document retrieval
- answer generation
- grounding verification
- abstention behavior
- a minimal interactive frontend

What it does not claim to be is a fully generalized finance QA platform. It is a focused RAG implementation with explicit safeguards and a clear extension path.
