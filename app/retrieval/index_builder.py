"""Build a reproducible FAISS index from Phase 2 processed paper JSON."""

import json
from pathlib import Path
from typing import Any

from app.retrieval.embeddings import EmbeddingModel, SentenceTransformerEmbedder
from app.retrieval.vector_store import DEFAULT_INDEX_PATH, FaissVectorStore

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "paper_id",
    "title",
    "authors",
    "section",
    "page",
    "text",
}


def load_processed_chunks(papers_dir: Path) -> list[dict[str, Any]]:
    """Read chunks from sorted Phase 2 JSON files without touching their PDFs."""
    paper_files = sorted(papers_dir.glob("*.json"))
    if not paper_files:
        raise FileNotFoundError(f"No processed paper JSON files found in {papers_dir}.")

    chunks: list[dict[str, Any]] = []
    for paper_file in paper_files:
        try:
            document = json.loads(paper_file.read_text(encoding="utf-8"))
            paper_chunks = document["chunks"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Could not read processed paper file {paper_file}: {error}") from error

        if not isinstance(paper_chunks, list):
            raise ValueError(f"The chunks field in {paper_file} must be a list.")
        for chunk in paper_chunks:
            if not isinstance(chunk, dict) or not REQUIRED_CHUNK_FIELDS.issubset(chunk):
                raise ValueError(f"A chunk in {paper_file} is missing required metadata.")
            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                continue
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No non-empty chunks found in processed papers under {papers_dir}.")
    return chunks


def build_index(
    papers_dir: Path = Path("data/papers"),
    index_path: Path = DEFAULT_INDEX_PATH,
    embedder: EmbeddingModel | None = None,
) -> int:
    """Embed all processed chunks and save a FAISS index and metadata sidecar."""
    chunks = load_processed_chunks(papers_dir)
    embedding_service = embedder or SentenceTransformerEmbedder()
    vectors = embedding_service.embed_documents([chunk["text"] for chunk in chunks])
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("The embedding service returned an unexpected vector matrix.")

    store = FaissVectorStore(dimension=int(vectors.shape[1]))
    store.add(vectors, chunks)
    store.save(index_path)
    return store.count