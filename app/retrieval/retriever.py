"""Query embedding and metadata-preserving vector retrieval."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import FaissVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A relevant chunk and its source metadata."""

    chunk_id: str
    paper_id: str
    title: str
    authors: list[str]
    section: str
    page: int
    text: str
    similarity_score: float
    retrieval_score: float | None = None
    retrieval_method: str = "dense"


class VectorRetriever:
    """Embed a question and retrieve the most similar indexed chunks."""

    def __init__(self, vector_store: FaissVectorStore, embedder: EmbeddingModel) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query text must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=str(result.metadata["chunk_id"]),
                paper_id=str(result.metadata["paper_id"]),
                title=str(result.metadata["title"]),
                authors=list(result.metadata["authors"]),
                section=str(result.metadata["section"]),
                page=int(result.metadata["page"]),
                text=str(result.metadata["text"]),
                similarity_score=result.similarity_score,
            )
            for result in results
        ]