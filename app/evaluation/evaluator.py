"""Deterministic local evaluation metrics for retrieval strategies."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_DATASET_PATH = Path("data/eval/attention_transformer_eval.json")
DEFAULT_PAPER_PATH = Path("data/papers/1706.03762v7.json")


class DatasetValidationError(ValueError):
    """Raised when the evaluation dataset does not match the ingested paper chunks."""


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics for one retrieval method at one cut-off K."""

    method: str
    K: int
    recall_at_k: float
    mrr_at_k: float
    average_latency: float


@dataclass
class EvaluationDataset:
    """Manually curated evaluation questions tied to real chunk IDs in the paper."""

    questions: list[dict[str, Any]]
    valid_chunk_ids: set[str]


def _load_valid_chunk_ids(paper_path: Path = DEFAULT_PAPER_PATH) -> set[str]:
    if not paper_path.exists():
        raise FileNotFoundError(f"Paper dataset not found: {paper_path}")

    payload = json.loads(paper_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise DatasetValidationError(f"Paper dataset is missing a valid chunk list: {paper_path}")
    valid_ids = {str(chunk["chunk_id"]) for chunk in chunks if isinstance(chunk, dict) and "chunk_id" in chunk}
    if not valid_ids:
        raise DatasetValidationError(f"No valid chunk IDs were found in {paper_path}")
    return valid_ids


def load_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    valid_chunk_ids: set[str] | None = None,
) -> EvaluationDataset:
    """Load and validate a local retrieval evaluation dataset."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise DatasetValidationError(f"Evaluation dataset should contain a JSON list of question entries: {path}")

    known_chunk_ids = valid_chunk_ids if valid_chunk_ids is not None else _load_valid_chunk_ids()
    validated: list[dict[str, Any]] = []
    for index, entry in enumerate(payload, start=1):
        if not isinstance(entry, Mapping):
            raise DatasetValidationError(f"Dataset item #{index} must be an object.")

        question = str(entry.get("question", "")).strip()
        if not question:
            raise DatasetValidationError(f"Dataset item #{index} is missing a non-empty question.")

        relevant_chunk_ids = entry.get("relevant_chunk_ids", [])
        if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
            raise DatasetValidationError(f"Dataset item #{index} must include at least one relevant_chunk_id.")

        cleaned_ids: list[str] = []
        for chunk_id in relevant_chunk_ids:
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise DatasetValidationError(f"Dataset item #{index} contains an invalid chunk_id: {chunk_id!r}")
            cleaned_id = chunk_id.strip()
            if cleaned_id not in known_chunk_ids:
                raise DatasetValidationError(
                    f"Dataset item #{index} references unknown chunk_id {cleaned_id!r}. "
                    "Only chunk IDs from the ingested paper are valid."
                )
            cleaned_ids.append(cleaned_id)

        question_entry: dict[str, Any] = {
            "question": question,
            "relevant_chunk_ids": cleaned_ids,
        }
        if "difficulty" in entry:
            question_entry["difficulty"] = entry.get("difficulty")
        if "category" in entry:
            question_entry["category"] = entry.get("category")
        validated.append(question_entry)

    return EvaluationDataset(questions=validated, valid_chunk_ids=set(known_chunk_ids))


def _extract_chunk_id(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        if "chunk_id" in result:
            return str(result["chunk_id"])
        if "id" in result:
            return str(result["id"])
    if hasattr(result, "chunk_id"):
        return str(result.chunk_id)
    raise TypeError(f"Unsupported ranked retrieval result type: {type(result)!r}")


def recall_at_k(ranked_chunk_ids: Sequence[str], relevant_chunk_ids: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not relevant_chunk_ids:
        return 0.0
    top_k_hits = set(ranked_chunk_ids[:k]) & relevant_chunk_ids
    return len(top_k_hits) / len(relevant_chunk_ids)


def mrr_at_k(ranked_chunk_ids: Sequence[str], relevant_chunk_ids: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not relevant_chunk_ids:
        return 0.0

    for rank, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def evaluate_queries(
    *,
    dataset: EvaluationDataset,
    retriever: Any,
    k_values: Sequence[int] = (1, 3, 5, 10),
    method_name: str = "retrieval",
) -> list[EvaluationResult]:
    """Evaluate one retrieval strategy against a local ground-truth dataset."""
    if not dataset.questions:
        raise ValueError("Evaluation dataset is empty.")

    results: list[EvaluationResult] = []
    for k in sorted(set(int(value) for value in k_values)):
        if k < 1:
            raise ValueError("All K values must be at least 1.")

        recall_total = 0.0
        mrr_total = 0.0
        latency_total = 0.0

        for question in dataset.questions:
            query = str(question["question"])
            relevant_ids = set(question["relevant_chunk_ids"])
            start = time.perf_counter()
            ranked = retriever.retrieve(query, top_k=max(k_values))
            latency_total += time.perf_counter() - start
            ordered_ids = [_extract_chunk_id(item) for item in ranked]
            recall_total += recall_at_k(ordered_ids, relevant_ids, k)
            mrr_total += mrr_at_k(ordered_ids, relevant_ids, k)

        question_count = len(dataset.questions)
        results.append(
            EvaluationResult(
                method=method_name,
                K=k,
                recall_at_k=recall_total / question_count,
                mrr_at_k=mrr_total / question_count,
                average_latency=latency_total / question_count,
            )
        )

    return results


def aggregate_results(results: Iterable[EvaluationResult]) -> dict[str, dict[int, dict[str, float]]]:
    """Average metrics by method and K across repeated evaluations."""
    grouped: dict[str, dict[int, list[EvaluationResult]]] = {}
    for result in results:
        grouped.setdefault(result.method, {}).setdefault(result.K, []).append(result)

    aggregated: dict[str, dict[int, dict[str, float]]] = {}
    for method_name, by_k in grouped.items():
        aggregated[method_name] = {}
        for k_value, evaluations in by_k.items():
            averaged = {
                "recall_at_k": sum(item.recall_at_k for item in evaluations) / len(evaluations),
                "mrr_at_k": sum(item.mrr_at_k for item in evaluations) / len(evaluations),
                "average_latency": sum(item.average_latency for item in evaluations) / len(evaluations),
            }
            aggregated[method_name][k_value] = averaged
    return aggregated
