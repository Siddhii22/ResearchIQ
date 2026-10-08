"""Deterministic local BM25 retrieval over processed paper chunks."""

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from app.retrieval.retriever import RetrievedChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase and extract ASCII alphanumeric tokens deterministically."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """A small Okapi BM25 index built in memory from Phase 2 chunk metadata."""

    def __init__(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        self._chunks = [dict(chunk) for chunk in chunks]
        self._tokens = [tokenize(str(chunk.get("text", ""))) for chunk in self._chunks]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._document_count = len(self._chunks)
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / self._document_count
            if self._document_count
            else 0.0
        )
        document_frequencies: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequencies.update(set(tokens))
        self._inverse_document_frequency = {
            token: math.log(
                1.0 + (self._document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in document_frequencies.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return matching chunks ordered by BM25 score with original metadata."""
        if not query.strip():
            raise ValueError("Query text must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query_tokens = tokenize(query)
        if not query_tokens or not self._chunks:
            return []

        scored_indices: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self._frequencies):
            document_length = len(self._tokens[index])
            score = 0.0
            for token in query_tokens:
                term_frequency = frequencies.get(token, 0)
                if term_frequency == 0:
                    continue
                inverse_frequency = self._inverse_document_frequency[token]
                length_normalizer = BM25_K1 * (
                    1.0 - BM25_B + BM25_B * document_length / max(self._average_length, 1.0)
                )
                score += inverse_frequency * (
                    term_frequency * (BM25_K1 + 1.0) / (term_frequency + length_normalizer)
                )
            if score > 0.0:
                scored_indices.append((index, score))

        scored_indices.sort(key=lambda item: (-item[1], item[0]))
        return [
            self._to_retrieved_chunk(self._chunks[index], score)
            for index, score in scored_indices[:top_k]
        ]

    @staticmethod
    def _to_retrieved_chunk(chunk: Mapping[str, Any], score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=str(chunk["chunk_id"]),
            paper_id=str(chunk["paper_id"]),
            title=str(chunk["title"]),
            authors=list(chunk["authors"]),
            section=str(chunk["section"]),
            page=int(chunk["page"]),
            text=str(chunk["text"]),
            similarity_score=0.0,
            retrieval_score=score,
            retrieval_method="bm25",
        )
