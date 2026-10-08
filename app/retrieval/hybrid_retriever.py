"""Dense plus BM25 retrieval with RRF fusion and cross-encoder reranking."""

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import ChunkReranker, CrossEncoderReranker
from app.retrieval.retriever import RetrievedChunk


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

    fused: dict[str, tuple[RetrievedChunk, float, int]] = {}
    next_order = 0
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            current = fused.get(chunk.chunk_id)
            if current is None:
                fused[chunk.chunk_id] = (chunk, 1.0 / (k + rank), next_order)
                next_order += 1
            else:
                original_chunk, score, order = current
                fused[chunk.chunk_id] = (original_chunk, score + 1.0 / (k + rank), order)

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

        dense_results = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)
        lexical_results = self.bm25_retriever.retrieve(query, top_k=self.bm25_top_k)
        fused = reciprocal_rank_fusion([dense_results, lexical_results], k=self.rrf_k)
        candidates = fused[: self.rerank_top_k]
        if not candidates:
            return []

        scores = self.reranker.score(query, [candidate.chunk for candidate in candidates])
        if len(scores) != len(candidates):
            raise ValueError("Reranker must return one score per candidate chunk.")

        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), -item[0].rrf_score),
        )
        return [
            replace(
                candidate.chunk,
                retrieval_score=float(score),
                retrieval_method="hybrid_rrf_reranked",
            )
            for candidate, score in reranked[:final_count]
        ]