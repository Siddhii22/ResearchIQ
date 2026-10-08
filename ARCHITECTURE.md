# ResearchIQ Architecture

## Intended Research Pipeline

```text
Research Question
        |
        v
     Planner
        |
        v
Literature Discovery
        |
        v
 Paper Ingestion
        |
        v
    Chunking
        |
        v
 Hybrid Retrieval
 (BM25 + dense search)
        |
        v
       RRF
        |
        v
    Reranking
        |
        v
 Evidence Analysis
        |
        v
      Critic <----- bounded revision loop
        |
        v
 Citation Verification
        |
        v
   Final Report
```

## Component Responsibilities

- **Planner:** translate the research question into focused search intents and an evidence plan.
- **Literature discovery:** find candidate papers from selected scholarly sources.
- **Paper ingestion:** retrieve and normalize paper metadata and content for processing.
- **Chunking:** split content along paper structure so retrieved passages retain their context.
- **Hybrid retrieval:** combine lexical BM25 search with dense semantic search to find candidate evidence.
- **RRF:** combine ranked candidate lists from different retrieval methods using Reciprocal Rank Fusion.
- **Reranking:** use a cross-encoder to refine the ordering of retrieved passages for the query.
- **Evidence analysis:** organize relevant passages and assess how they support the research question.
- **Critic:** inspect the draft for unsupported claims, missing evidence, and incomplete coverage.
- **Citation verification:** check that report citations resolve to retrieved sources and support the associated claims.
- **Final report:** present a structured synthesis with evidence-linked citations and clear limitations.

## Agent Workflow

The agentic workflow is planned for LangGraph. Nodes will represent bounded research steps, and explicit state will carry the question, discovered sources, retrieved evidence, draft, and validation feedback. The critic may request a limited number of revisions; a fixed retry bound prevents an unending loop. The workflow will then either return a verified report or clearly report that the evidence is insufficient.

## Phase 1 Boundary

Phase 1 implements only the FastAPI application foundation, environment-backed settings, and `GET /health`. Literature search, parsing, retrieval, LLM calls, LangGraph agents, citation checks, report generation, and the frontend are planned work, not current behavior.