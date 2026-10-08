import json
from pathlib import Path

import pytest

from app.evaluation.evaluator import (
    EvaluationDataset,
    EvaluationResult,
    DatasetValidationError,
    aggregate_results,
    evaluate_queries,
    load_dataset,
    mrr_at_k,
    recall_at_k,
)


class FakeRetriever:
    def __init__(self, ordered_ids_by_query: dict[str, list[str]]):
        self.ordered_ids_by_query = ordered_ids_by_query

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        return self.ordered_ids_by_query.get(query, [])[:top_k]


def test_recall_at_k_counts_hits_in_top_k() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}

    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1.0


def test_mrr_at_k_returns_first_relevant_rank() -> None:
    ranked = ["x", "b", "y", "d"]
    relevant = {"b", "d"}

    assert mrr_at_k(ranked, relevant, 1) == 0.0
    assert mrr_at_k(ranked, relevant, 2) == 0.5
    assert mrr_at_k(ranked, relevant, 4) == 0.5


def test_dataset_validation_rejects_unknown_chunk_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bad_dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is the Transformer?",
                    "relevant_chunk_ids": ["missing-chunk-id"],
                    "difficulty": "easy",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError):
        load_dataset(dataset_path, valid_chunk_ids={"real-chunk"})


def test_result_aggregation_computes_average_scores() -> None:
    results = [
        EvaluationResult(method="dense", K=1, recall_at_k=1.0, mrr_at_k=1.0, average_latency=0.01),
        EvaluationResult(method="dense", K=1, recall_at_k=0.5, mrr_at_k=0.5, average_latency=0.02),
        EvaluationResult(method="bm25", K=3, recall_at_k=0.75, mrr_at_k=0.25, average_latency=0.03),
    ]

    aggregated = aggregate_results(results)

    assert aggregated["dense"][1]["recall_at_k"] == pytest.approx(0.75)
    assert aggregated["dense"][1]["mrr_at_k"] == pytest.approx(0.75)
    assert aggregated["bm25"][3]["average_latency"] == pytest.approx(0.03)


def test_evaluate_queries_works_for_deterministic_mock_retrieval() -> None:
    dataset = EvaluationDataset(
        questions=[
            {
                "question": "What does the model use instead of recurrence?",
                "relevant_chunk_ids": ["chunk-a", "chunk-b"],
                "difficulty": "easy",
                "category": "motivation",
            }
        ],
        valid_chunk_ids={"chunk-a", "chunk-b", "chunk-c"},
    )

    fake_retriever = FakeRetriever(
        {
            "What does the model use instead of recurrence?": ["chunk-c", "chunk-a", "chunk-b"],
        }
    )

    results = evaluate_queries(
        dataset=dataset,
        retriever=fake_retriever,
        k_values=(1, 2, 3),
        method_name="dense",
    )

    assert len(results) == 3
    assert results[0].method == "dense"
    assert results[0].K == 1
    assert results[0].recall_at_k == 0.0
    assert results[1].K == 2
    assert results[1].recall_at_k == 0.5
    assert results[2].K == 3
    assert results[2].mrr_at_k == pytest.approx(0.5)
