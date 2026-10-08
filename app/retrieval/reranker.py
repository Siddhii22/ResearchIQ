"""Lazy CPU cross-encoder scoring for retrieved chunk candidates."""

from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np

from app.config import settings
from app.retrieval.retriever import RetrievedChunk


class ChunkReranker(Protocol):
    def score(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        """Return one relevance score per candidate, in input order."""


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


class CrossEncoderReranker:
    """Load the configured cross-encoder only when its first scoring call occurs."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.rerank_model

    def score(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        if not query.strip():
            raise ValueError("Query text must not be empty.")
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = _load_cross_encoder(self.model_name).predict(pairs, show_progress_bar=False)
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if values.size != len(chunks) or not np.isfinite(values).all():
            raise ValueError("The cross-encoder returned invalid candidate scores.")
        return [float(score) for score in values]