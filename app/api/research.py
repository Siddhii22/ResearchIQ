"""Research query endpoint backed by the local vector index and LLM service."""

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import (
    ResearchQueryRequest,
    ResearchQueryResponse,
    ResearchSourceResponse,
)
from app.config import settings
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import VectorRetriever
from app.retrieval.vector_store import DEFAULT_INDEX_PATH, FaissVectorStore
from app.services.llm_service import LLMConfigurationError, LLMRequestError, create_llm_service
from app.services.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/api/v1/research", tags=["research"])


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    try:
        vector_store = FaissVectorStore.load(DEFAULT_INDEX_PATH)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="The local vector index is unavailable. Run scripts/build_index.py first.",
        ) from error

    dense_retriever = VectorRetriever(vector_store, SentenceTransformerEmbedder())
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=BM25Retriever(vector_store.chunks),
        reranker=CrossEncoderReranker(),
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        rrf_k=settings.rrf_k,
        rerank_top_k=settings.rerank_top_k,
        final_top_k=settings.final_top_k,
    )
    return RAGPipeline(
        retriever=hybrid_retriever,
        llm=create_llm_service(),
    )


@router.post("/query", response_model=ResearchQueryResponse)
def research_query(
    request: ResearchQueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ResearchQueryResponse:
    try:
        result = pipeline.answer_question(request.question, top_k=request.top_k)
    except LLMConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except LLMRequestError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return ResearchQueryResponse(
        answer=result.answer,
        sources=[
            ResearchSourceResponse(
                citation_id=f"[{source.citation_id}]",
                chunk_id=source.chunk_id,
                paper_id=source.paper_id,
                title=source.title,
                authors=source.authors,
                section=source.section,
                page=source.page,
                text=source.text,
                similarity_score=source.similarity_score,
                retrieval_score=source.retrieval_score,
                retrieval_method=source.retrieval_method,
            )
            for source in result.sources
        ],
    )