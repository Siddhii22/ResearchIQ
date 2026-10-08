"""Local FAISS inner-product index with a JSON chunk-metadata sidecar."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import faiss
import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_INDEX_PATH = Path("data/index/researchiq.faiss")


@dataclass(frozen=True)
class VectorSearchResult:
    """A matched chunk's original metadata and cosine similarity score."""

    metadata: dict[str, Any]
    similarity_score: float


class FaissVectorStore:
    """Exact cosine-similarity search using normalized FAISS inner products."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("Vector dimension must be positive.")
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[dict[str, Any]]:
        """Return copies of all chunk metadata aligned with the indexed vectors."""
        return [dict(chunk) for chunk in self._chunks]

    @staticmethod
    def metadata_path_for(index_path: Path) -> Path:
        return index_path.with_suffix(".json")

    def add(self, embeddings: ArrayLike, chunks: Sequence[Mapping[str, Any]]) -> None:
        """Add vectors and aligned chunk metadata to the index."""
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional matrix.")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}.")
        if vectors.shape[0] != len(chunks):
            raise ValueError("The number of vectors must match the number of chunks.")
        if not chunks:
            return
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values.")

        vectors = np.ascontiguousarray(vectors.copy())
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("Zero-length embeddings cannot be indexed.")
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._chunks.extend(dict(chunk) for chunk in chunks)

    def search(self, query_embedding: ArrayLike, top_k: int = 5) -> list[VectorSearchResult]:
        """Return up to ``top_k`` chunk records ordered by cosine similarity."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not self._chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape != (1, self.dimension):
            raise ValueError(f"Expected a query vector with dimension {self.dimension}.")
        if not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError("Query embedding must be finite and non-zero.")

        query = np.ascontiguousarray(query.copy())
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(top_k, self.count))
        return [
            VectorSearchResult(
                metadata=self._chunks[int(index)],
                similarity_score=float(score),
            )
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]

    def save(self, index_path: Path = DEFAULT_INDEX_PATH) -> None:
        """Persist the FAISS index and aligned chunk metadata sidecar."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = self.metadata_path_for(index_path)
        temporary_index = index_path.with_name(f"{index_path.name}.tmp")
        temporary_metadata = metadata_path.with_name(f"{metadata_path.name}.tmp")
        try:
            faiss.write_index(self._index, str(temporary_index))
            temporary_metadata.write_text(
                json.dumps(
                    {"dimension": self.dimension, "chunks": self._chunks},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_index, index_path)
            os.replace(temporary_metadata, metadata_path)
        finally:
            temporary_index.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)

    @classmethod
    def load(cls, index_path: Path = DEFAULT_INDEX_PATH) -> "FaissVectorStore":
        """Load a persisted index after checking its metadata alignment."""
        metadata_path = cls.metadata_path_for(index_path)
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"FAISS index or metadata sidecar is missing for {index_path}.")

        metadata_document = json.loads(metadata_path.read_text(encoding="utf-8"))
        dimension = int(metadata_document["dimension"])
        chunks = metadata_document["chunks"]
        index = faiss.read_index(str(index_path))
        if index.d != dimension or index.ntotal != len(chunks):
            raise ValueError("FAISS index and metadata sidecar do not match.")

        store = cls(dimension)
        store._index = index
        store._chunks = chunks
        return store