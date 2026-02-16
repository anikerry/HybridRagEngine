# Hybrid-RAG Knowledge Engine

A fully local, privacy-first Hybrid Retrieval-Augmented Generation (RAG)
system designed for navigating dense technical documentation with high
accuracy on domain-specific terminology.

This project combines semantic vector search and keyword-based retrieval
using Reciprocal Rank Fusion (RRF), and supports local LLM inference via
Ollama for fully offline deployment.

------------------------------------------------------------------------

## 🚀 Overview

Traditional vector-based RAG systems often struggle with:

-   Domain-specific acronyms
-   Exact technical terminology
-   Constraint-heavy documentation
-   Regulatory text search
-   Precise keyword matching

Pure semantic similarity can miss critical exact-match content.\
This system addresses that limitation using hybrid retrieval.

------------------------------------------------------------------------

## 🧠 Core Idea

Instead of relying solely on embeddings, this system:

1.  Performs semantic vector search (Qdrant + BGE embeddings)
2.  Performs keyword search (BM25)
3.  Fuses rankings using Reciprocal Rank Fusion (RRF)
4.  Assembles top-ranked context
5.  Generates grounded answers using a local LLM
6.  Enforces structured JSON output with citations

------------------------------------------------------------------------

## 🏗 System Architecture

User Query\
→ Vector Search (Qdrant)\
→ BM25 Keyword Search\
→ Reciprocal Rank Fusion (RRF)\
→ Context Assembly\
→ Local LLM (Ollama)\
→ JSON Output with Citations

------------------------------------------------------------------------

## 🔧 Technology Stack

-   Python 3.10+
-   LlamaIndex
-   Qdrant (Vector Database)
-   rank-bm25
-   HuggingFace Embeddings (BAAI/bge-small-en-v1.5)
-   Ollama (Llama3 / Mistral / Phi3)
-   Docker + WSL2

Fully local. No external API dependency required.

------------------------------------------------------------------------

## 📂 Project Structure

hybrid_rag_engine/ │ ├── data/ │ └── docs/ \# Source PDFs / Markdown
files │ ├── storage/ \# BM25 storage artifacts │ ├── src/ │ ├──
ingest.py \# Document ingestion + indexing │ ├── ask.py \# Hybrid
retrieval + generation │ ├── hybrid.py \# BM25 + RRF implementation │
└── config.py \# Configuration parameters │ ├── docker-compose.yml \#
Qdrant container setup ├── requirements.txt └── README.md

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone Repository

git clone `<repo-url>`{=html}\
cd hybrid_rag_engine

------------------------------------------------------------------------

### 2. Create Environment

Using Conda:

conda create -n genai_env python=3.10\
conda activate genai_env\
pip install -r requirements.txt

------------------------------------------------------------------------

### 3. Start Vector Database (Qdrant)

docker compose up -d

Verify service:

curl http://localhost:6333

------------------------------------------------------------------------

### 4. Install Ollama (Local LLM)

Download from: https://ollama.com

Pull a model:

ollama pull mistral\
or\
ollama pull llama3

------------------------------------------------------------------------

## 📥 Document Ingestion

Place source files inside:

data/docs/

Supported formats: - PDF - Markdown

Run ingestion:

python src/ingest.py

------------------------------------------------------------------------

## 🔎 Query the System

python src/ask.py

Example:

Ask: What are the compliance requirements?

------------------------------------------------------------------------

## 🔍 Hybrid Retrieval Logic

Vector Search\
- Semantic similarity\
- Captures paraphrasing\
- Uses BGE embeddings

BM25\
- Exact keyword matching\
- Handles acronyms and precise constraints

Reciprocal Rank Fusion (RRF)

Score = Σ 1 / (k + rank)

------------------------------------------------------------------------

## 📊 Design Advantages

-   Handles technical jargon better than vector-only RAG
-   Reduces hallucination risk
-   Supports offline inference
-   GPU acceleration supported via Ollama
-   Modular and extensible architecture
-   Citation-enforced outputs

------------------------------------------------------------------------

## 🧪 Example Use Cases

-   Engineering documentation search
-   Regulatory compliance lookup
-   Academic policy assistant
-   API documentation navigator
-   Internal knowledge assistant

------------------------------------------------------------------------

## 📈 Future Improvements

-   Cross-encoder reranker
-   Query expansion
-   Evaluation harness
-   Streamlit UI
-   Multi-collection support
-   Embedding caching
-   Latency metrics

------------------------------------------------------------------------

## 🛡 Design Principles

-   Privacy-first
-   Fully local execution
-   Deterministic retrieval
-   Transparent citations
-   Modular architecture

------------------------------------------------------------------------

## 📄 License

MIT License

------------------------------------------------------------------------

## 👤 Author

AI Engineering Project -- Hybrid Retrieval Systems Exploration
