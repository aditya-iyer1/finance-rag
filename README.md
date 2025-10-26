# Hallucination-Resistant Finance RAG

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to answer questions over real-world financial filings (e.g., 10-Ks), with an emphasis on hallucination resistance and response grounding.

## Overview

The system parses raw SEC filings, segments and embeds them using dense retrieval, and queries them via a language model to generate grounded answers. It includes tools for claim attribution, hallucination detection, and interface components for evaluation and interactive QA.

## Key Features

- PDF section parsing and chunking for 10-K filings  
- Sentence-transformer embeddings stored in persistent ChromaDB  
- Top-k vector retrieval for grounding context  
- LangChain-based QA chains (OpenAI / OpenRouter compatible)  
- Modular support for response generation, evaluation, and UI  
- Built-in tools for scoring factual consistency and token attribution  

## Directory Structure

```
hallucination-resistant-finance-rag/
│
├── data/                    # Raw PDFs and persistent ChromaDB
│   ├── raw_pdfs/
│   └── chroma_index/
│
├── rag_pipeline/           # Core application code
│   ├── parser/             # PDF loading, section parsing, chunking
│   ├── retriever/          # Embedding, vector DB storage & querying
│   ├── llm/                # QA chain construction and generation logic
│   └── evaluation/         # Grounding, hallucination, attribution tools
│
├── streamlit_app/          # (Planned) Streamlit frontend for QA interface
│
├── notebooks/              # Jupyter notebooks for testing and experimentation
│
├── tests/                  # (Optional) Unit and integration tests
│
├── embed_chunks_cli.py     # CLI script for end-to-end embedding
├── requirements.txt        # Dependency list
├── README.md               # Project overview and usage guide
└── .gitignore              # Git exclusions
```

## Setup Instructions

```bash
# Clone repository
git clone https://github.com/aditya-iyer1/finance-rag.git
cd finance-rag

# Create and activate virtual environment
python -m venv rag
source rag/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```
