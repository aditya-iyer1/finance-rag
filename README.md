# Hallucination-Resistant Finance RAG

A production-oriented RAG pipeline for answering questions over SEC financial filings (10-Ks) with built-in safeguards against hallucination. The system uses intent-based hybrid retrieval, traceable citations, and confidence-based abstention to ensure answers are grounded in source documents.

## Definitive Summary

### What it does

This project answers questions about SEC filings, such as 10-Ks, and tries hard not to make things up. It stores multiple filings in one index, but every query is explicitly scoped to one active document so retrieval and generation never mix companies. A minimal Streamlit app is included so you can index filings, choose the active one, ask a question, and inspect the retrieved context.

### Architecture

```
┌─────────────────┐
│  PDF Document   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  INGESTION                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │  Parse   │→ │ Section  │→ │  Chunk   │→ │ Embed   ││
│  │   PDF    │  │  Split   │  │  Text    │  │  Store  ││
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘│
│                    (ITEM 1,      (token-      (ChromaDB)│
│                     2, 1A...)    aware)                │
└─────────────────────────────────────────────────────────┘
         │
         │ Metadata: doc_id, section, chunk_id
         ▼
┌─────────────────────────────────────────────────────────┐
│  QUERY & RETRIEVAL                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Classify    │→ │   Semantic   │→ │   Keyword    │ │
│  │   Intent     │  │    Search    │  │   Fallback   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  (HQ_LOCATION,      (vector sim)      (if poor)        │
│   INCORPORATION,                                        │
│   etc.)                                                 │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  GENERATION & VERIFICATION                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Format     │→ │   LLM        │→ │  Grounding   │ │
│  │   Prompt     │  │  Generate    │  │   Check      │ │
│  │  (citations) │  │   Answer     │  │  + Gate      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
         │
         ▼
    Final Answer
    (with citations or abstain)
```

### Key design decisions with tradeoffs

**Why Hybrid Retrieval?** Pure semantic search can miss relevant chunks when embeddings do not capture domain-specific relationships, such as "headquartered" versus "based." Keyword fallback improves recall, but it adds some extra logic and makes ranking behavior less uniform than a pure vector-only pipeline.

**Why Metadata?** Chunk-level metadata (`doc_id`, `section`, `chunk_id`) makes answers traceable and debuggable. The tradeoff is a little more ingestion complexity and the need to keep metadata consistent across parsing, storage, retrieval, logging, and UI display.

**Why LLM-as-Judge for Grounding?** A second model pass can catch errors that simple word overlap misses, such as wrong locations, wrong figures, or confident paraphrases that are not actually supported by the source text. The tradeoff is added latency and cost, plus a dependency on another model call; this repo keeps a word-overlap fallback so the system still degrades gracefully if the judge fails.

**Why Structured JSON for Abstention?** The answer-generation step returns structured JSON with an `is_answerable` flag and an `answer` field, which makes abstention handling more reliable than string-matching free-form prose. The tradeoff is tighter coupling to model output format and slightly more prompt complexity, but it is much easier to reason about programmatically.

**Why Active-Document Scoping?** Multiple filings can be stored in one shared index, but every query is filtered to one explicit `active_doc_id`. This avoids cross-company retrieval and mixed citations. The tradeoff is that once more than one filing is indexed, callers must specify which filing they want to query.

### How to run

**1. Setup**

```bash
python -m venv rag
source rag/bin/activate  # On Windows: rag\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
```

**2. Index a filing**

```bash
python embed_chunks_cli.py
```

The CLI scans `data/raw-pdfs/`, lets you choose a PDF, and writes its chunks to ChromaDB. Multiple filings can live in the same index at once. In normal use, indexing is a one-time cost per document: once a filing has been embedded and stored, you can switch between indexed documents without re-embedding. If you re-index a filing that is already present, only that filing's chunks are replaced.

**3. Query from Python or validation scripts**

```python
from rag_pipeline.retriever.retrieve import query_chunks
from rag_pipeline.llm.generator import generate_answer_with_gate

chunks = query_chunks("Where is the company headquartered?", top_k=5, active_doc_id="tesla-2024-10K")
answer, abstained = generate_answer_with_gate("Where is the company headquartered?", chunks)
print(answer, abstained)
```

Or run the bundled validation matrices:

```bash
python run_validation.py
python run_validation_jpmc.py
```

**4. Query from Streamlit**

```bash
streamlit run app.py
```

The Streamlit UI shows the available raw PDFs, lets you choose the active indexed filing, and shows the answer with citations plus a collapsible retrieved-context panel for transparency.

You must index at least one filing before the app can answer queries.

### Limitations & what I'd do next

**Current limitations**

- The embedding model is lightweight and general-purpose, not finance-specific.
- The system uses one embedding model for both indexing and retrieval, with no task-specific or document-specific embedding strategy.
- Retrieval does not yet use a reranker, so the top-k set can still include near-misses.
- Multiple filings can be stored together, but every query still requires one active document scope.
- Intent classification is rule-based, so new query styles may require synonym updates.
- Page-level sourcing is not preserved in metadata, so citations resolve to section and chunk rather than exact PDF page.
- LLM outputs are not perfectly deterministic in practice, even with `temperature=0`, so edge-case answers and abstentions can vary slightly across runs.
- Queries with significant typos may cause abstention since both the embedding model and keyword matcher expect reasonable spelling.
- LLM-based answering and grounding checks add latency and cost.

**What I’d do next**

- Add cross-encoder reranking before generation.
- Add caching and cost/latency instrumentation around both the answer model and the grounding judge.
- Expand evaluation from test matrices to finer-grained citation and claim-level checks.
- Add stronger document-level permissions and document routing beyond the current active-doc filter.
- Upgrade embeddings or compare domain-specific alternatives for finance-heavy queries.

## Key Features

- **Traceability & Citations**: Every answer includes numbered citations `[1]`, `[2]` mapping to specific document sections with metadata (doc_id, section, chunk_id)
- **Confidence-Based Abstention**: System abstains with clear reasoning when retrieval quality is poor, context is insufficient, or verification fails
- **Intent-Based Hybrid Retrieval**: Combines semantic vector search with keyword fallback, using rule-based intent classification (HQ_LOCATION, INCORPORATION, BUSINESS_OVERVIEW, FINANCIALS_REVENUE, RISKS, AUDITOR) to prioritize relevant sections
- **Active-Document Query Scoping**: Multiple filings can share one index, but retrieval is always filtered to one explicit `active_doc_id`
- **Single-Document Mode Awareness**: When only one document exists in the index, entity terms are downweighted so they do not dominate ranking
- **Evidence Pattern Matching**: Uses regex patterns to detect strong signals (e.g., "headquartered in Austin, Texas") for improved retrieval quality
- **Logging & Observability**: Structured Python `logging` (no bare `print()`) with `debug=True` for intent classification, retrieval scores, chunk metadata, and confidence decisions
- **Modular Pipeline**: Clean separation of concerns (parser → retriever → generator → verifier) for easy extension and testing

## System Architecture

### Core Modules

**Ingestion Pipeline** (`rag_pipeline/parser/`, `rag_pipeline/retriever/embed_store.py`):
1. **PDF Parsing**: Extracts raw text from PDFs, splits by "ITEM" patterns to identify sections (ITEM 1, ITEM 2, ITEM 1A, etc.)
2. **Chunking**: Token-aware chunking with overlap using `tiktoken`, preserving section boundaries
3. **Embedding**: Sentence-transformer embeddings (`paraphrase-MiniLM-L3-v2`) for semantic search
4. **Storage**: Persistent ChromaDB with metadata (doc_id, section, chunk_id) for traceability

**Query Pipeline** (`rag_pipeline/retriever/hybrid_retrieve.py`, `rag_pipeline/retriever/intent_classifier.py`, `rag_pipeline/retriever/chroma_client.py`):
1. **Active Document Resolution**: Validates the requested `active_doc_id` and refuses ambiguous multi-document queries
2. **Document Filtering**: Applies `doc_id` filtering before semantic retrieval and keyword fallback so a query only sees one filing
3. **Intent Classification**: Rule-based classification using synonym matching (6 intent types)
4. **Semantic Retrieval**: Vector similarity search using query embeddings
5. **Intent Coverage Check**: Evaluates if semantic results contain intent keywords or evidence patterns
6. **Keyword Fallback**: Full-corpus keyword search if semantic results lack intent coverage
7. **Scoring**: `10×intent_matches + 3×evidence_patterns + entity_weight×entity_matches + length_bonus`
8. **Tiny Chunk Filtering**: Removes chunks < 200 chars before final selection

**Generation** (`rag_pipeline/llm/generator.py`):
1. **Prompt Formatting**: Creates prompt with numbered context chunks and citation instructions
2. **Structured JSON Output**: LLM returns `{"is_answerable": bool, "answer": str}` for reliable programmatic abstention detection
3. **LLM Call**: GPT-4o with temperature=0 and `response_format: json_object` for deterministic, parseable answers
4. **Logging**: Records query, used chunks (with metadata), and answer length via Python `logging` module

**Verification** (`rag_pipeline/verifier/confidence_gate.py`, `rag_pipeline/verifier/hallucination_guard.py`):
1. **Pre-generation Gate**: Checks if chunks retrieved (abstains if empty)
2. **LLM Abstention Detection**: Parses structured JSON `is_answerable` field from the generation response
3. **Post-generation Gate**: Validates chunk count, context size, intent evidence, and answer grounding
4. **Grounding Check (LLM-as-a-Judge)**: Uses `gpt-4o-mini` to verify factual entailment between the answer and retrieved chunks, with a word-overlap fallback if the judge call fails
5. **Abstention**: Returns clear abstain message with reason if any check fails

## Behind the Scenes: Design Choices

**Why Hybrid Retrieval?** Pure semantic search can miss relevant chunks when embeddings don't capture domain-specific relationships (e.g., "headquartered" vs "based"). Keyword fallback ensures recall when semantic search fails, especially for location/incorporation queries where exact phrase matching matters.

**Why Metadata?** Chunk-level metadata (doc_id, section, chunk_id) enables traceability: users can verify answers by checking source sections, and the system can log which chunks were used for debugging and evaluation.

**Why LLM-as-a-Judge for Grounding?** After LLM generation, the system uses a second model pass (`gpt-4o-mini`) to verify that the answer is factually entailed by the retrieved chunks. This catches subtle grounding failures that simple lexical checks can miss, but it introduces extra latency and cost. If the judge call fails, the system falls back to a basic word-overlap check.

**Why Structured JSON for Abstention?** The generation prompt requires the model to return `{"is_answerable": bool, "answer": str}` so the application can distinguish answerable versus unanswerable cases without fragile string matching. This improves reliability, but it also means the generation step depends on consistent structured output from the model.

**Why Active-Document Querying?** Storing multiple filings in one index avoids repeated embedding work, but unscoped queries would risk mixing companies in retrieval. Requiring an explicit active document keeps answers and citations confined to one filing at the cost of one extra parameter when more than one document is indexed.

**Intent-Based Scoring:** Instead of relying solely on vector similarity (which can be noisy), the system scores chunks by intent keyword matches and evidence patterns. This prioritizes chunks that actually contain relevant information (e.g., ITEM 2 for HQ questions) over semantically similar but irrelevant chunks.

**Single-Document Mode:** When only one document exists in the index, entity terms (e.g., "Tesla") appear in nearly every chunk, making them poor discriminators. The system automatically detects single-doc mode and downweights entity matches (0.1x vs 0.5x) to prioritize intent signals.

## Repository Structure

```
hallucination-resistant-finance-rag/
│
├── data/                           # Data storage
│   ├── raw-pdfs/                  # Input PDF files
│   └── chroma_index/              # Persistent ChromaDB vector store
│
├── rag_pipeline/                  # Core application code
│   ├── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── ingest/
│   │   └── embed.py
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py         # PDF text extraction, section parsing
│   │   └── chunker.py            # Token-aware chunking with overlap
│   │
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── chroma_client.py      # Unified ChromaDB client and active-doc resolution
│   │   ├── embed_store.py        # Embedding computation & ChromaDB storage
│   │   ├── retrieve.py           # Entry point: query_chunks() → hybrid_retrieve()
│   │   ├── hybrid_retrieve.py    # Intent-based hybrid retrieval (semantic + keyword)
│   │   └── intent_classifier.py  # Rule-based intent classification (6 types)
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── generator.py          # Prompt formatting, LLM generation, logging
│   │
│   └── verifier/
│       ├── confidence_gate.py    # Pre/post-generation confidence checks
│       └── hallucination_guard.py # LLM-as-a-Judge grounding verification (gpt-4o-mini)
│
├── notebooks/                     # Jupyter notebooks for testing
│   ├── 01_pdf_section_parser.ipynb
│   ├── 02_chunking_test.ipynb
│   ├── 03_retrieval_test.ipynb
│   ├── 04_generation_test.ipynb  # Interactive QA interface
│   ├── 05_walkthrough.ipynb      # End-to-end walkthrough with edge case tests
│   └── notebook_init.py          # Required to initialize notebooks in correct environment
│
├── app.py                        # Minimal Streamlit frontend
├── embed_chunks_cli.py           # CLI script: PDF → embeddings → ChromaDB
├── run_validation.py             # Validation matrix for Tesla-focused checks
├── run_validation_jpmc.py        # Validation matrix for JPMorgan Chase-focused checks
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## How to Run

### Environment Setup

```bash
# Create virtual environment
python -m venv rag
source rag/bin/activate  # On Windows: rag\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenAI key:

```bash
OPENAI_API_KEY=your_key_here
```

### Indexing Documents

```bash
# Launch the interactive indexer (select a PDF from data/raw-pdfs/)
python embed_chunks_cli.py

# With verbose output (section list, metadata samples, progress bar)
python embed_chunks_cli.py --verbose
```

The CLI is interactive:

1. It scans `data/raw-pdfs/` for available 10-K PDFs.
2. It shows a numbered menu so you can choose which filing to index.
3. It parses the selected PDF into ITEM sections, chunks the text, embeds those chunks, and stores them in ChromaDB.
4. If the selected document is already indexed, it asks whether you want to re-index just that document.
5. Existing chunks for other indexed documents are left untouched.

This means the workflow after running `embed_chunks_cli.py` is file-selection driven rather than hardcoded to a single source document. You can keep Tesla and JPMorgan Chase in the same index, then switch query targets by selecting a different active document in the UI or by passing `active_doc_id` in code. Switching between already indexed documents does not require re-embedding.

### Querying

**Option 0: Streamlit frontend**
```bash
streamlit run app.py
```

The app loads `OPENAI_API_KEY` from `.env`, shows the available PDFs in `data/raw-pdfs/`, lets you select the active indexed filing, and lets you ask one question at a time while inspecting retrieved chunk metadata in the sidebar.

**Option 1: Jupyter Notebook** (recommended for testing)
```bash
jupyter notebook notebooks/04_generation_test.ipynb
```

**Option 2: Python Script**
```python
from rag_pipeline.retriever.retrieve import query_chunks
from rag_pipeline.llm.generator import generate_answer_with_gate

chunks = query_chunks("Where is the company headquartered?", top_k=5, active_doc_id="jpmc-10k-2025")
answer, abstained = generate_answer_with_gate("Where is the company headquartered?", chunks)
print(f"Answer: {answer}")
print(f"Abstained: {abstained}")
```

If more than one filing is indexed, omitting `active_doc_id` raises an error rather than risking cross-document retrieval.

## Testing & Evaluation

### Validation Scripts

The repo includes two validation entry points that exercise the same retrieval and answer-generation pipeline against different indexed filings:

```bash
# Tesla-focused validation matrix
python run_validation.py

# JPMorgan Chase-focused validation matrix
python run_validation_jpmc.py
```

Use `--debug` with either script to surface retrieval logs and `--no-color` for plain ASCII output.

To test against a specific filing:

1. Run `python embed_chunks_cli.py`.
2. Select the Tesla or JPMorgan Chase 10-K from `data/raw-pdfs/`.
3. If that filing is already indexed, let the CLI replace only that filing's chunks if prompted.
4. Run the matching validation script for that indexed document.

The Tesla matrix checks Tesla-specific known-answer and abstention cases. The JPMorgan Chase matrix does the same for JPMC, including questions like headquarters, business overview, auditor, risk factors, and incorporation, plus out-of-scope prompts that should abstain.

### Query Test Matrix

| Category | Example Query | Success Criteria |
|----------|--------------|------------------|
| **HQ Location** | "Where is the company headquartered?" | Retrieves ITEM 2. PROPERTIES; answers with location; includes citation `[1]` |
| **Incorporation** | "What state is the company incorporated in?" | Retrieves legal/corporate section; answers with state (e.g., "Delaware"); no confusion with HQ |
| **Business Overview** | "What does the company do?" | Retrieves ITEM 1. BUSINESS; provides high-level summary; no abstain |
| **Financials** | "What was the company's revenue last year?" | Retrieves financial statements; answers with citation OR abstains if data missing |
| **Risks** | "What are the company's major risks?" | Retrieves ITEM 1A. RISK FACTORS; detailed answer; no hallucination |
| **Auditor** | "Who audits the company?" | Retrieves Exhibit/Item 14; correct firm name (e.g., "PricewaterhouseCoopers LLP") |
| **Abstention** | "What are the company's plans for nuclear energy?" | Detects intent; retrieval fails; system abstains with clear reason |

### What "Success" Looks Like

- **Citations Present**: Answers include numbered citations `[1]`, `[2]` mapping to source sections
- **Abstains When Missing**: System abstains gracefully when information is unavailable
- **No Hallucination**: Answers are grounded in retrieved chunks (verified via LLM-as-a-Judge entailment check)
- **Relevant Retrieval**: Top chunks come from appropriate sections (ITEM 2 for HQ, ITEM 1A for risks, etc.)
- **Intent Classification**: System correctly identifies query intent (HQ_LOCATION, INCORPORATION, etc.)

## Example

**Query:** "Where is the company headquartered?"

**Retrieved Chunks:**
- `[1]` ITEM 2. PROPERTIES: "We are headquartered in Austin, Texas. Our principal facilities include..."

**Answer:**
```
The company is headquartered in Austin, Texas [1]. The company's principal facilities 
include properties in North America, Europe, and Asia utilized for manufacturing, 
warehousing, engineering, retail and service locations, and administrative offices [1].
```

**System Logs** (with `debug=True`):
```
rag_pipeline.retriever.hybrid_retrieve - DEBUG - Detected intents: ['HQ_LOCATION'], confidence: 0.45
rag_pipeline.retriever.hybrid_retrieve - DEBUG - Active document scope: tesla-2024-10K
rag_pipeline.retriever.hybrid_retrieve - DEBUG - Semantic coverage: has_coverage=True, intent_matches=1, evidence_hits=1
rag_pipeline.retriever.hybrid_retrieve - DEBUG - Final result_chunks count=1
rag_pipeline.llm.generator - INFO - Query: Where is the company headquartered?
rag_pipeline.llm.generator - INFO -   [1] tesla-2024-10K::ITEM 2. PROPERTIES::chunk-71 (distance=19.9896)
```

## Limitations & Future Improvements

**Current Limitations:**
- **Embedding Model**: Uses `paraphrase-MiniLM-L3-v2` (384-dim), which may not capture domain-specific financial terminology optimally
- **Single Embedding Strategy**: The same embedding model is used for all indexed filings and all queries; there is no per-document or task-specific embedding selection
- **No Reranking**: Top-k retrieval without cross-encoder reranking can miss subtle relevance signals
- **No Page-Level Citations**: Metadata does not currently preserve PDF page numbers, so citations resolve to section and chunk rather than exact page
- **Rule-Based Intent**: Intent classification is rule-based; may miss nuanced queries or require manual synonym updates
- **Typo Sensitivity**: Queries with significant typos may cause abstention since both the embedding model and keyword matcher expect reasonable spelling
- **Residual Non-Determinism**: Even with `temperature=0`, model outputs and abstention behavior can vary slightly across runs
- **LLM-as-a-Judge Cost**: Grounding verification uses a `gpt-4o-mini` call per answer, adding latency and cost; a local NLI model could reduce this
- **Frontend Scope Is Minimal**: The Streamlit app is intentionally narrow and does not yet support indexing, multi-document workflows, or richer citation browsing

**Planned Improvements:**
- **Reranker Integration**: Add cross-encoder reranking (e.g., `ms-marco-MiniLM`) to refine top-k results
- **Better Evaluators**: Implement claim-level attribution, fact-checking against source, and automated evaluation metrics
- **Multi-Document Policy**: Add document-level permissions, filtering policies, and richer multi-doc workflow controls on top of the current active-doc filter
- **Cost Control**: Add caching for repeated queries, token usage tracking, and cost-aware model selection
- **UI/Frontend**: Expand the Streamlit app to support indexing actions, richer source inspection, and better document switching
- **Domain-Specific Embeddings**: Fine-tune or use financial-domain embeddings (e.g., FinBERT) for better semantic understanding

## Status

This project now includes a fully working core pipeline plus a minimal Streamlit entrypoint for interactive querying. The system is still intentionally narrow in scope, but the ingestion, retrieval, generation, verification, validation scripts, and frontend are all wired together end to end. The modular architecture supports future upgrades such as rerankers, multi-document support, richer evaluation, and a more capable UI.
