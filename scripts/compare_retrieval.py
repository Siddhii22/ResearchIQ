"""Compare dense-only, BM25-only, RRF, and reranked retrieval for one query."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import VectorRetriever
from app.retrieval.vector_store import DEFAULT_INDEX_PATH, FaissVectorStore


def print_results(label: str, rows: list[tuple[str, float]]) -> None:
    print(f"\n{label}")
    if not rows:
        print("  no matching chunks")
        return
    for rank, (chunk_id, score) in enumerate(rows, start=1):
        print(f"  {rank}. {chunk_id}  score={score:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Research question to compare across retrieval modes")
    parser.add_argument("--top-k", type=int, default=settings.final_top_k)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    arguments = parser.parse_args()
    if arguments.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        store = FaissVectorStore.load(arguments.index_path)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Could not load vector index: {error}", file=sys.stderr)
        return 1

    dense = VectorRetriever(store, SentenceTransformerEmbedder())
    bm25 = BM25Retriever(store.chunks)
    dense_results = dense.retrieve(arguments.query, top_k=settings.dense_top_k)
    bm25_results = bm25.retrieve(arguments.query, top_k=settings.bm25_top_k)
    fused_results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        k=settings.rrf_k,
    )
    hybrid = HybridRetriever(
        dense,
        bm25,
        CrossEncoderReranker(settings.rerank_model),
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        rrf_k=settings.rrf_k,
        rerank_top_k=settings.rerank_top_k,
        final_top_k=arguments.top_k,
    )
    reranked_results = hybrid.retrieve(arguments.query, top_k=arguments.top_k)

    print_results(
        "Dense-only (FAISS cosine similarity)",
        [(chunk.chunk_id, chunk.similarity_score) for chunk in dense_results[: arguments.top_k]],
    )
    print_results(
        "BM25-only",
        [(chunk.chunk_id, float(chunk.retrieval_score or 0.0)) for chunk in bm25_results[: arguments.top_k]],
    )
    print_results(
        "Hybrid RRF",
        [(item.chunk.chunk_id, item.rrf_score) for item in fused_results[: arguments.top_k]],
    )
    print_results(
        "Hybrid RRF + cross-encoder",
        [(chunk.chunk_id, float(chunk.retrieval_score or 0.0)) for chunk in reranked_results],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
