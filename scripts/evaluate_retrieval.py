"""Evaluate local retrieval strategies against a ground-truth dataset for the Transformer paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.evaluation.evaluator import EvaluationDataset, aggregate_results, evaluate_queries, load_dataset
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import VectorRetriever
from app.retrieval.vector_store import DEFAULT_INDEX_PATH, FaissVectorStore


class RRFOnlyRetriever:
    def __init__(self, dense_retriever, bm25_retriever, *, dense_top_k: int, bm25_top_k: int, rrf_k: int) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5):
        dense_results = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=self.bm25_top_k)
        fused = reciprocal_rank_fusion([dense_results, bm25_results], k=self.rrf_k)
        return [item.chunk for item in fused[:top_k]]


def _print_table(results: list[dict[str, object]]) -> None:
    headers = ("method", "K", "Recall@K", "MRR@K", "avg_latency(s)")
    rows = [
        (
            str(row["method"]),
            str(row["K"]),
            f"{float(row['Recall@K']):.4f}",
            f"{float(row['MRR@K']):.4f}",
            f"{float(row['average_latency']):.6f}",
        )
        for row in results
    ]
    widths = [max(len(header), max(len(value) for value in column)) for header, column in zip(headers, zip(*([headers] + rows)), strict=False)]
    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "  ".join("-" * width for width in widths)
    print(header_line)
    print(divider)
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/attention_transformer_eval.json"))
    parser.add_argument("--k", type=int, nargs="*", default=[1, 3, 5, 10])
    parser.add_argument("--save-json", action="store_true", help="Save the evaluation results as JSON under data/eval/results/")
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval/results"))
    arguments = parser.parse_args()

    if not arguments.k:
        parser.error("At least one K value is required.")

    try:
        dataset = load_dataset(arguments.dataset)
    except FileNotFoundError as error:
        print(f"Could not load evaluation dataset: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Invalid evaluation dataset: {error}", file=sys.stderr)
        return 1

    try:
        store = FaissVectorStore.load(DEFAULT_INDEX_PATH)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Could not load the local FAISS index. Run scripts/build_index.py first: {error}", file=sys.stderr)
        return 1

    dense_retriever = VectorRetriever(store, SentenceTransformerEmbedder())
    bm25_retriever = BM25Retriever(store.chunks)
    rrf_retriever = RRFOnlyRetriever(
        dense_retriever,
        bm25_retriever,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        rrf_k=settings.rrf_k,
    )
    hybrid_retriever = HybridRetriever(
        dense_retriever,
        bm25_retriever,
        CrossEncoderReranker(settings.rerank_model),
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        rrf_k=settings.rrf_k,
        rerank_top_k=settings.rerank_top_k,
        final_top_k=max(arguments.k),
    )

    strategy_retrievers = {
        "dense": dense_retriever,
        "bm25": bm25_retriever,
        "hybrid_rrf": rrf_retriever,
        "hybrid_rrf_reranker": hybrid_retriever,
    }

    all_results: list[dict[str, object]] = []
    for method_name, retriever in strategy_retrievers.items():
        eval_rows = evaluate_queries(
            dataset=dataset,
            retriever=retriever,
            k_values=tuple(arguments.k),
            method_name=method_name,
        )
        for row in eval_rows:
            all_results.append(
                {
                    "method": row.method,
                    "K": row.K,
                    "Recall@K": row.recall_at_k,
                    "MRR@K": row.mrr_at_k,
                    "average_latency": row.average_latency,
                }
            )

    print("\nRetrieval evaluation on the local Transformer dataset\n")
    _print_table(all_results)

    if arguments.save_json:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = arguments.output_dir / "retrieval_evaluation_results.json"
        output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nSaved JSON results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
