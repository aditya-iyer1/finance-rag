# Hallucination-Resistant Finance RAG

A Retrieval-Augmented Generation (RAG) system that extracts and answers questions about real-world financial documents (e.g., 10-Ks) — with hallucination detection and grounding evaluation.

## Features

- ✅ PDF Parsing for 10-Ks and earnings reports
- ✅ Embedding & Chunking with ChromaDB + SentenceTransformers
- ✅ LangChain-based QA pipeline with OpenAI / OpenRouter
- ✅ Streamlit frontend with citations & hallucination score
- ✅ Evaluation tools for grounding, token attribution, and claim checking

## Project Structure

```plaintext
hallucination-resistant-finance-rag/
│
├── data/                    # Raw and processed financial documents
│   └── raw_pdfs/
│
├── rag_pipeline/           # Core logic
│   ├── parser/             # PDF loading and chunking
│   ├── retriever/          # Embedding and vector DB
│   ├── llm/                # LangChain QA chains
│   └── evaluation/         # Grounding, scoring, attribution
│
├── streamlit_app/          # Interactive UI
│
├── notebooks/              # Experiments, eval visualization
│
├── tests/                  # Unit tests for pipeline modules
│
├── requirements.txt        # Dependencies
├── README.md               # Project overview
└── .gitignore              # Git exclusions
```

## Setup

```bash
# Clone the repo
git clone git@github.com:aditya-iyer1/finance-rag.git
cd finance-rag

# Create virtual environment
python -m venv rag
source rag/bin/activate

# Install dependencies
pip install -r requirements.txt
```