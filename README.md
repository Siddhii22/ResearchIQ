# ResearchIQ

## Overview
ResearchIQ is an evidence-driven AI research assistant project. Its goal is to help users move from a research question to a structured report whose claims can be traced to relevant literature.

## Problem
Reliable AI-assisted research requires more than generating fluent text. Relevant literature must be discovered and organized, evidence must be retrieved from the source material, citations must support the claims they accompany, and unsupported statements must be caught before a report is presented. Without these checks, a language model can omit important work, cite irrelevant sources, or hallucinate details.

## Planned Architecture
The planned pipeline starts with a research question and literature discovery, ingests papers, and divides them into structure-aware chunks. It combines lexical and dense retrieval, fuses candidate rankings, reranks evidence, then synthesizes and checks claims against citations. A critic will identify gaps and request bounded revisions before a final report is returned. The LangGraph workflow, retrieval, synthesis, and verification steps remain future work.

## Tech Stack
**Implemented:** Python, FastAPI, Uvicorn, python-dotenv, HTTPX, pytest, and pypdf. Phase 2 uses the arXiv Atom API for paper metadata and HTTPX for paper PDF downloads.

**Planned for later phases:** sentence-transformers for dense embeddings, BM25 for lexical retrieval, a vector database, Reciprocal Rank Fusion (RRF), a cross-encoder reranker, an LLM API, and LangGraph for the bounded agent workflow. These planned components are not installed or active yet.

## Current Status
Phase 2 — Paper ingestion. The project can fetch arXiv metadata, download and cache a PDF, extract page text, create section-aware chunks with source metadata, and save a local JSON representation. Phase 1's API foundation remains in place. Retrieval, generation, agent, and frontend functionality are not implemented.

## Project Structure
```text
app/
  api/          HTTP routes
  agents/       Reserved for later agent workflow
  ingestion/    arXiv metadata, PDF processing, chunking, and JSON storage
  retrieval/    Reserved for later evidence retrieval
  evaluation/   Reserved for later evaluation tools
  models/       Reserved for later application models
  services/     Reserved for later application services
  utils/        Reserved for shared utilities
  config.py     Environment-backed settings
  main.py       FastAPI application
data/
  papers/       Local paper data (ignored by Git)
  eval/         Local evaluation data (ignored by Git)
tests/          Automated tests
frontend/       Reserved; no UI is implemented yet
scripts/        Project scripts, including the ingestion demo
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

The `.env` file is optional for the current API and ingestion pipeline. `APP_NAME` sets the API title. `LLM_API_KEY` is a placeholder for a later phase; no LLM calls are made now.

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

## Testing
Run the test suite from the project root:

```powershell
pytest
```

## Future Roadmap
1. Add support for additional literature sources and improve paper structure extraction.
2. Add BM25 and dense retrieval, then rank fusion and cross-encoder reranking.
3. Build evidence-grounded synthesis and citation verification.
4. Implement the bounded critic-and-revision workflow with LangGraph.
5. Add evaluation datasets and a professional research-workspace UI.