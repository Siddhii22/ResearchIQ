# ResearchIQ

## Overview
ResearchIQ is an evidence-driven AI research assistant project. Its goal is to help users move from a research question to a structured report whose claims can be traced to relevant literature.

## Problem
Reliable AI-assisted research requires more than generating fluent text. Relevant literature must be discovered and organized, evidence must be retrieved from the source material, citations must support the claims they accompany, and unsupported statements must be caught before a report is presented. Without these checks, a language model can omit important work, cite irrelevant sources, or hallucinate details.

## Planned Architecture
The planned pipeline starts with a research question and literature discovery, ingests papers, and divides them into structure-aware chunks. The current baseline uses dense vector retrieval; later versions can add lexical retrieval, rank fusion, reranking, and a critic that identifies gaps and requests bounded revisions. The LangGraph workflow, hybrid retrieval, and automated citation verification remain future work.

## Tech Stack
**Implemented:** Python, FastAPI, Uvicorn, python-dotenv, HTTPX, pytest, pypdf, sentence-transformers, NumPy, FAISS CPU, and the official `google-genai` SDK. Phase 2 uses the arXiv Atom API and PDF text extraction. Phase 3 uses `sentence-transformers/all-MiniLM-L6-v2`, a local FAISS index, and Gemini text generation by default. Phase 4 combines local BM25, RRF, and `cross-encoder/ms-marco-MiniLM-L-6-v2`; an OpenAI-compatible LLM provider remains optional.

**Planned for later phases:** BM25 for lexical retrieval, hybrid retrieval, Reciprocal Rank Fusion (RRF), a cross-encoder reranker, and LangGraph for the bounded agent workflow. These components are not installed or active yet.

## Current Status
Phase 4 — Hybrid retrieval. The project can retrieve evidence through dense FAISS and BM25, fuse rankings with configurable RRF, and rerank candidates with a lazy CPU cross-encoder before grounded generation. No formal retrieval evaluation, agent workflow, or frontend is implemented.

## Project Structure
```text
app/
  api/          HTTP routes
  agents/       Reserved for later agent workflow
  ingestion/    arXiv metadata, PDF processing, chunking, and JSON storage
  retrieval/    MiniLM embeddings, FAISS, BM25, RRF, and cross-encoder reranking
  evaluation/   Reserved for later evaluation tools
  models/       Reserved for later application models
  services/     Configurable LLM client and RAG orchestration
  utils/        Reserved for shared utilities
  config.py     Environment-backed settings
  main.py       FastAPI application
data/
  papers/       Local paper data (ignored by Git)
  eval/         Local evaluation data (ignored by Git)
tests/          Automated tests
frontend/       Reserved; no UI is implemented yet
scripts/        Ingestion and vector-index build scripts
```

## Setup
Use Python 3.10 or newer. From the project root, create and activate a virtual environment in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

The `.env` file is optional for the health endpoint and ingestion pipeline. `APP_NAME` sets the API title. Set `LLM_PROVIDER=gemini`, `GEMINI_API_KEY`, and `GEMINI_MODEL` in `.env` for Gemini generation. No credentials are included in the repository.

## Running Locally
With the virtual environment active, start the API from the project root:

```powershell
uvicorn app.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health`. To run the API with Docker Compose, first create `.env` from `.env.example`, then run `docker compose up --build`.

To ingest an arXiv paper, run this from the project root:

```powershell
python scripts/ingest_paper.py 2401.12345
```

The script writes the cached PDF and processed JSON under `data/papers/`. It does not require an API key.

Build or refresh the local vector index from processed paper JSON (this downloads the configured embedding model the first time):

```powershell
python scripts/build_index.py
```

The FAISS index and chunk metadata are stored under `data/index/` and are ignored by Git.

The research query endpoint now uses hybrid retrieval. Configure `DENSE_TOP_K`, `BM25_TOP_K`, `RRF_K`, `RERANK_TOP_K`, `FINAL_TOP_K`, and `RERANK_MODEL` in `.env` to adjust the pipeline. The first query that reaches reranking downloads the cross-encoder model if it is not cached.

Compare dense-only, BM25-only, RRF, and RRF plus reranking using the same query:

```powershell
python scripts/compare_retrieval.py "What is the main contribution of the Transformer architecture?"
```

This reports ranked chunk IDs and scores only; it does not calculate retrieval accuracy metrics.

To verify Gemini configuration and model access with one short request, run:

```powershell
python scripts/check_gemini.py
```

The diagnostic prints only key-presence and sanitized success/failure status, never the key or provider response. To use the optional OpenAI-compatible API instead, set `LLM_PROVIDER=openai_compatible`, `LLM_API_KEY`, `LLM_API_BASE_URL`, and `LLM_MODEL`. Without the selected provider's key, `/api/v1/research/query` returns HTTP 503 rather than a fabricated answer.

Example request:

```powershell
$body = @{ question = "What is the main contribution of the Transformer architecture?"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/research/query" -Method Post -ContentType "application/json" -Body $body
```

## Testing
Run the test suite from the project root:

```powershell
pytest
```

## Future Roadmap
1. Add BM25 and hybrid retrieval, then rank fusion and cross-encoder reranking.
2. Improve citation verification and evidence quality checks.
3. Implement the bounded critic-and-revision workflow with LangGraph.
4. Add evaluation datasets and a professional research-workspace UI.