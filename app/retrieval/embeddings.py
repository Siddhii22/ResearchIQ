"""CPU sentence-transformer embeddings for chunks and research queries."""

from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from app.config import settings

FloatVector = NDArray[np.float32]


class EmbeddingModel(Protocol):
    """Interface shared by the index builder and retriever."""

    def embed_documents(self, texts: Sequence[str]) -> FloatVector:
        """Return one normalized vector per input text."""
        ...

    def embed_query(self, query: str) -> FloatVector:
        """Return a normalized vector for a single query."""
        ...


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


class SentenceTransformerEmbedder:
    """Load a sentence-transformers model once and return normalized vectors."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model

    def embed_documents(self, texts: Sequence[str]) -> FloatVector:
        if not texts:
            raise ValueError("At least one document is required for embedding.")
        if any(not text.strip() for text in texts):
            raise ValueError("Document text must not be empty.")

        vectors = _load_model(self.model_name).encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError("The embedding model returned an unexpected document vector shape.")
        return np.ascontiguousarray(matrix)

    def embed_query(self, query: str) -> FloatVector:
        if not query.strip():
            raise ValueError("Query text must not be empty.")
        return self.embed_documents([query])[0]