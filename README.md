# Hallucination-Resistant Finance RAG

A production-oriented RAG pipeline for answering questions over SEC financial filings (10-Ks) with built-in safeguards against hallucination. The system uses intent-based hybrid retrieval, traceable citations, and confidence-based abstention to ensure answers are grounded in source documents.

## Summary

This project implements a Retrieval-Augmented Generation (RAG) system designed to answer questions over financial filings with high fidelity. Unlike naive RAG implementations that may hallucinate or provide ungrounded answers, this system includes multiple layers of verification: intent-based retrieval that prioritizes relevant sections, chunk-level metadata for traceability, post-generation grounding checks, and confidence-based abstention when information is missing or retrieval quality is poor. The architecture is modular, extensible, and designed to work across multiple question types (HQ location, incorporation, business overview, financials, risks, auditor) without query-specific hardcoding.

## Key Features

- **Traceability & Citations**: Every answer includes numbered citations `[1]`, `[2]` mapping to specific document sections with metadata (doc_id, section, chunk_id)
- **Confidence-Based Abstention**: System abstains with clear reasoning when retrieval quality is poor, context is insufficient, or verification fails
- **Intent-Based Hybrid Retrieval**: Combines semantic vector search with keyword fallback, using rule-based intent classification (HQ_LOCATION, INCORPORATION, BUSINESS_OVERVIEW, FINANCIALS_REVENUE, RISKS, AUDITOR) to prioritize relevant sections
- **Single-Document Mode Awareness**: Automatically downweights entity terms when only one document exists, preventing entity matches from dominating ranking
- **Evidence Pattern Matching**: Uses regex patterns to detect strong signals (e.g., "headquartered in Austin, Texas") for improved retrieval quality
- **Logging & Observability**: Structured Python `logging` (no bare `print()`) with `debug=True` for intent classification, retrieval scores, chunk metadata, and confidence decisions
- **Modular Pipeline**: Clean separation of concerns (parser → retriever → generator → verifier) for easy extension and testing

## System Architecture

### Pipeline Flow

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
│  │   Intent     │  │    Search    │  │   Fallback  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  (HQ_LOCATION,      (vector sim)      (if poor)        │
│   INCORPORATION,                                        │
│   etc.)                                                 │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  GENERATION & VERIFICATION                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Format     │→ │   LLM        │→ │  Grounding  │ │
│  │   Prompt     │  │  Generate    │  │   Check     │ │
│  │  (citations) │  │   Answer      │  │  + Gate     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
         │
         ▼
    Final Answer
    (with citations or abstain)
```

### Core Modules

**Ingestion Pipeline** (`rag_pipeline/parser/`, `rag_pipeline/retriever/embed_store.py`):
1. **PDF Parsing**: Extracts raw text from PDFs, splits by "ITEM" patterns to identify sections (ITEM 1, ITEM 2, ITEM 1A, etc.)
2. **Chunking**: Token-aware chunking with overlap using `tiktoken`, preserving section boundaries
3. **Embedding**: Sentence-transformer embeddings (`paraphrase-MiniLM-L3-v2`) for semantic search
4. **Storage**: Persistent ChromaDB with metadata (doc_id, section, chunk_id) for traceability

**Query Pipeline** (`rag_pipeline/retriever/hybrid_retrieve.py`, `rag_pipeline/retriever/intent_classifier.py`):
1. **Intent Classification**: Rule-based classification using synonym matching (6 intent types)
2. **Semantic Retrieval**: Vector similarity search using query embeddings
3. **Intent Coverage Check**: Evaluates if semantic results contain intent keywords or evidence patterns
4. **Keyword Fallback**: Full-corpus keyword search if semantic results lack intent coverage
5. **Scoring**: `10×intent_matches + 3×evidence_patterns + entity_weight×entity_matches + length_bonus`
6. **Tiny Chunk Filtering**: Removes chunks < 200 chars before final selection

**Generation** (`rag_pipeline/llm/generator.py`):
1. **Prompt Formatting**: Creates prompt with numbered context chunks and citation instructions
2. **Structured JSON Output**: LLM returns `{"is_answerable": bool, "answer": str}` for reliable programmatic abstention detection
3. **LLM Call**: GPT-4o with temperature=0 and `response_format: json_object` for deterministic, parseable answers
4. **Logging**: Records query, used chunks (with metadata), and answer length via Python `logging` module

**Verification** (`rag_pipeline/verifier/confidence_gate.py`, `rag_pipeline/verifier/hallucination_guard.py`):
1. **Pre-generation Gate**: Checks if chunks retrieved (abstains if empty)
2. **LLM Abstention Detection**: Parses structured JSON `is_answerable` field from the generation response
3. **Post-generation Gate**: Validates chunk count, context size, intent evidence, and answer grounding
4. **Grounding Check (LLM-as-a-Judge)**: Uses `gpt-4o-mini` to verify factual entailment between the answer and retrieved chunks, catching subtle hallucinations (e.g., correct structure but wrong location/numbers) that lexical overlap would miss
5. **Abstention**: Returns clear abstain message with reason if any check fails

## Behind the Scenes: Design Choices

**Why Hybrid Retrieval?** Pure semantic search can miss relevant chunks when embeddings don't capture domain-specific relationships (e.g., "headquartered" vs "based"). Keyword fallback ensures recall when semantic search fails, especially for location/incorporation queries where exact phrase matching matters.

**Why Metadata?** Chunk-level metadata (doc_id, section, chunk_id) enables traceability: users can verify answers by checking source sections, and the system can log which chunks were used for debugging and evaluation.

**How Grounding Verification Works:** After LLM generation, the system uses an **LLM-as-a-Judge** approach (`gpt-4o-mini`) to verify that the answer is factually entailed by the retrieved chunks. The judge prompt explicitly handles coreference ("We"/"Our" in filings mapping to the company name in the answer) and checks for contradictions in locations, figures, and dates. If the judge determines the answer is not grounded, the system abstains. A word-overlap fallback is used if the judge API call fails.

**Intent-Based Scoring:** Instead of relying solely on vector similarity (which can be noisy), the system scores chunks by intent keyword matches and evidence patterns. This prioritizes chunks that actually contain relevant information (e.g., ITEM 2 for HQ questions) over semantically similar but irrelevant chunks.

**Single-Document Mode:** When only one document exists, entity terms (e.g., "Tesla") appear in nearly every chunk, making them poor discriminators. The system automatically detects single-doc mode and downweights entity matches (0.1x vs 0.5x) to prioritize intent signals.

## Repository Structure

```
hallucination-resistant-finance-rag/
│
├── data/                           # Data storage
│   ├── raw-pdfs/                  # Input PDF files
│   └── chroma_index/              # Persistent ChromaDB vector store
│
├── rag_pipeline/                  # Core application code
│   ├── parser/
│   │   ├── pdf_loader.py         # PDF text extraction, section parsing
│   │   └── chunker.py            # Token-aware chunking with overlap
│   │
│   ├── retriever/
│   │   ├── chroma_client.py      # Unified ChromaDB client (single-doc detection)
│   │   ├── embed_store.py        # Embedding computation & ChromaDB storage
│   │   ├── retrieve.py           # Entry point: query_chunks() → hybrid_retrieve()
│   │   ├── hybrid_retrieve.py    # Intent-based hybrid retrieval (semantic + keyword)
│   │   └── intent_classifier.py  # Rule-based intent classification (6 types)
│   │
│   ├── llm/
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
├── embed_chunks_cli.py           # CLI script: PDF → embeddings → ChromaDB
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

### Indexing Documents

```bash
# Index the default PDF (data/raw-pdfs/tesla-2024-10K.pdf → data/chroma_index/)
python embed_chunks_cli.py

# With verbose output (section list, metadata samples, progress bar)
python embed_chunks_cli.py --verbose
```

> **Note:** The PDF path is currently set in `embed_chunks_cli.py`. Edit the `pdf_path` variable to index a different filing.

### Querying

**Option 1: Jupyter Notebook** (recommended for testing)
```bash
jupyter notebook notebooks/04_generation_test.ipynb
```

**Option 2: Python Script**
```python
from rag_pipeline.retriever.retrieve import query_chunks
from rag_pipeline.llm.generator import generate_answer_with_gate

chunks = query_chunks("Where is the company headquartered?", top_k=5)
answer, abstained = generate_answer_with_gate("Where is the company headquartered?", chunks)
print(f"Answer: {answer}")
print(f"Abstained: {abstained}")
```

## Testing & Evaluation

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
rag_pipeline.retriever.hybrid_retrieve - DEBUG - Single-doc mode: True
rag_pipeline.retriever.hybrid_retrieve - DEBUG - Semantic coverage: has_coverage=True, intent_matches=1, evidence_hits=1
rag_pipeline.retriever.hybrid_retrieve - DEBUG - Final result_chunks count=1
rag_pipeline.llm.generator - INFO - Query: Where is the company headquartered?
rag_pipeline.llm.generator - INFO -   [1] tesla-2024-10K::ITEM 2. PROPERTIES::chunk-71 (distance=19.9896)
```

## Limitations & Future Improvements

**Current Limitations:**
- **Embedding Model**: Uses `paraphrase-MiniLM-L3-v2` (384-dim), which may not capture domain-specific financial terminology optimally
- **No Reranking**: Top-k retrieval without cross-encoder reranking can miss subtle relevance signals
- **Single-Document Focus**: Optimized for single 10-K queries; multi-document scenarios may need permission/access control
- **Rule-Based Intent**: Intent classification is rule-based; may miss nuanced queries or require manual synonym updates
- **LLM-as-a-Judge Cost**: Grounding verification uses a `gpt-4o-mini` call per answer, adding latency and cost; a local NLI model could reduce this
- **No Frontend**: Core pipeline is implemented, but UI/frontend is work-in-progress

**Planned Improvements:**
- **Reranker Integration**: Add cross-encoder reranking (e.g., `ms-marco-MiniLM`) to refine top-k results
- **Better Evaluators**: Implement claim-level attribution, fact-checking against source, and automated evaluation metrics
- **Multi-Document Support**: Add document-level permissions, filtering, and multi-doc query routing
- **Cost Control**: Add caching for repeated queries, token usage tracking, and cost-aware model selection
- **UI/Frontend**: Build Streamlit interface for interactive QA with citation visualization
- **Domain-Specific Embeddings**: Fine-tune or use financial-domain embeddings (e.g., FinBERT) for better semantic understanding

## Status

This is a **framework/project scaffold** with a fully implemented core pipeline. The ingestion, retrieval, generation, and verification modules are production-ready, but the frontend UI is work-in-progress. The system is designed to be extensible: new intent types can be added to `intent_classifier.py`, and the modular architecture supports easy integration of rerankers, evaluators, and UI components.
