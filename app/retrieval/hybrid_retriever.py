"""Dense plus BM25 retrieval with RRF fusion and cross-encoder reranking."""

import logging
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Protocol, Sequence, TypeVar

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import ChunkReranker, CrossEncoderReranker
from app.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)
_Result = TypeVar("_Result")
RERANKER_RELEVANCE_THRESHOLD = 0.0


def _run_timed_stage(stage: str, operation: Callable[[], _Result]) -> _Result:
    started = perf_counter()
    try:
        result = operation()
    except Exception as error:
        logger.warning(
            "research_stage_failed stage=%s elapsed_seconds=%.3f error_type=%s",
            stage,
            perf_counter() - started,
            type(error).__name__,
        )
        raise
    logger.info(
        "research_stage_completed stage=%s elapsed_seconds=%.3f",
        stage,
        perf_counter() - started,
    )
    return result


class RankedRetriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return chunks in descending relevance order."""


@dataclass(frozen=True)
class FusedChunk:
    chunk: RetrievedChunk
    rrf_score: float


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    k: int = 60,
) -> list[FusedChunk]:
    """Combine ranked lists using standard 1-based Reciprocal Rank Fusion."""
    if k < 1:
        raise ValueError("RRF k must be at least 1.")

    fused: dict[tuple[str, str], tuple[RetrievedChunk, float, int]] = {}
    next_order = 0
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            source_key = (chunk.paper_id, chunk.chunk_id)
            current = fused.get(source_key)
            if current is None:
                fused[source_key] = (chunk, 1.0 / (k + rank), next_order)
                next_order += 1
            else:
                original_chunk, score, order = current
                fused[source_key] = (original_chunk, score + 1.0 / (k + rank), order)

    ordered = sorted(fused.values(), key=lambda item: (-item[1], item[2]))
    return [FusedChunk(chunk=chunk, rrf_score=score) for chunk, score, _ in ordered]


class HybridRetriever:
    """Retrieve dense and lexical candidates, fuse, rerank, and return final top-K."""

    def __init__(
        self,
        dense_retriever: RankedRetriever,
        bm25_retriever: BM25Retriever,
        reranker: ChunkReranker | None = None,
        *,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        rerank_top_k: int = 20,
        final_top_k: int = 5,
    ) -> None:
        if min(dense_top_k, bm25_top_k, rerank_top_k, final_top_k) < 1:
            raise ValueError("Retrieval candidate counts must be at least 1.")
        if rrf_k < 1:
            raise ValueError("RRF k must be at least 1.")
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query text must not be empty.")
        final_count = self.final_top_k if top_k is None else top_k
        if final_count < 1:
            raise ValueError("top_k must be at least 1.")

        dense_results = _run_timed_stage(
            "dense_retrieval",
            lambda: self.dense_retriever.retrieve(query, top_k=self.dense_top_k),
        )
        lexical_results = _run_timed_stage(
            "bm25_retrieval",
            lambda: self.bm25_retriever.retrieve(query, top_k=self.bm25_top_k),
        )
        fused = reciprocal_rank_fusion([dense_results, lexical_results], k=self.rrf_k)
        candidates = fused[: self.rerank_top_k]
        if not candidates:
            return []

        scores = _run_timed_stage(
            "cross_encoder_reranking",
            lambda: self.reranker.score(query, [candidate.chunk for candidate in candidates]),
        )
        if len(scores) != len(candidates):
            raise ValueError("Reranker must return one score per candidate chunk.")

        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), -item[0].rrf_score),
        )
        relevant = [
            (candidate, float(score))
            for candidate, score in reranked
            if float(score) > RERANKER_RELEVANCE_THRESHOLD
        ]
        if not relevant:
            return []

        diverse: list[tuple[FusedChunk, float]] = []
        remaining: list[tuple[FusedChunk, float]] = []
        seen_papers: set[str] = set()
        for candidate, score in relevant:
            if candidate.chunk.paper_id in seen_papers:
                remaining.append((candidate, score))
                continue
            seen_papers.add(candidate.chunk.paper_id)
            diverse.append((candidate, score))
        selected = (diverse + remaining)[:final_count]
        return [
            replace(
                candidate.chunk,
                retrieval_score=float(score),
                retrieval_method="hybrid_rrf_reranked",
            )
            for candidate, score in selected
        ]