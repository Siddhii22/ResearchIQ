
"""Build a FAISS index from processed paper JSON using small batches."""

import json
from pathlib import Path
from typing import Any, Iterator

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

# Keep each embedding operation small for low-memory deployments.
EMBEDDING_BATCH_SIZE = 8


def iter_processed_chunks(
    papers_dir: Path,
) -> Iterator[dict[str, Any]]:
    """Yield validated chunks from sorted processed paper JSON files."""
    paper_files = sorted(papers_dir.glob("*.json"))
    if not paper_files:
        raise FileNotFoundError(
            f"No processed paper JSON files found in {papers_dir}."
        )

    found_chunk = False

    for paper_file in paper_files:
        try:
            document = json.loads(paper_file.read_text(encoding="utf-8"))
            paper_chunks = document["chunks"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"Could not read processed paper file {paper_file}: {error}"
            ) from error

        if not isinstance(paper_chunks, list):
            raise ValueError(
                f"The chunks field in {paper_file} must be a list."
            )

        for chunk in paper_chunks:
            if (
                not isinstance(chunk, dict)
                or not REQUIRED_CHUNK_FIELDS.issubset(chunk)
            ):
                raise ValueError(
                    f"A chunk in {paper_file} is missing required metadata."
                )

            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                continue

            found_chunk = True
            yield chunk

    if not found_chunk:
        raise ValueError(
            f"No non-empty chunks found in processed papers under {papers_dir}."
        )


def load_processed_chunks(papers_dir: Path) -> list[dict[str, Any]]:
    """Read all validated chunks into memory for callers that need a list."""
    return list(iter_processed_chunks(papers_dir))


def _batches(items, batch_size: int):
    """Yield lists containing at most batch_size items."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def build_index(
    papers_dir: Path = Path("data/papers"),
    index_path: Path = DEFAULT_INDEX_PATH,
    embedder: EmbeddingModel | None = None,
) -> int:
    """Build and save the FAISS index in small embedding batches."""
    embedding_service = embedder or SentenceTransformerEmbedder()
    store = None
    total = 0

    for chunk_batch in _batches(
        iter_processed_chunks(papers_dir),
        EMBEDDING_BATCH_SIZE,
    ):
        vectors = embedding_service.embed_documents(
            [chunk["text"] for chunk in chunk_batch]
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(chunk_batch):
            raise ValueError(
                "The embedding service returned an unexpected vector matrix."
            )

        if store is None:
            store = FaissVectorStore(dimension=int(vectors.shape[1]))

        store.add(vectors, chunk_batch)
        total += len(chunk_batch)

        # Persist progress after every batch. If the process stops, the
        # next request can still load a valid index containing completed batches.
        store.save(index_path)

    if store is None or total == 0:
        raise ValueError(
            f"No non-empty chunks found in processed papers under {papers_dir}."
        )

    return total


def append_processed_chunks(
    papers_dir: Path = Path("data/papers"),
    index_path: Path = DEFAULT_INDEX_PATH,
    embedder: EmbeddingModel | None = None,
) -> int:
    """Append unindexed chunks in small batches and save after each batch."""
    try:
        store = FaissVectorStore.load(index_path)
    except FileNotFoundError:
        return build_index(papers_dir, index_path, embedder)

    indexed_keys = {
        (str(chunk.get("paper_id", "")), str(chunk.get("chunk_id", "")))
        for chunk in store.chunks
    }

    new_chunks = (
        chunk
        for chunk in iter_processed_chunks(papers_dir)
        if (
            str(chunk["paper_id"]),
            str(chunk["chunk_id"]),
        ) not in indexed_keys
    )

    embedding_service = embedder or SentenceTransformerEmbedder()

    for chunk_batch in _batches(new_chunks, EMBEDDING_BATCH_SIZE):
        vectors = embedding_service.embed_documents(
            [chunk["text"] for chunk in chunk_batch]
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(chunk_batch):
            raise ValueError(
                "The embedding service returned an unexpected vector matrix."
            )

        if int(vectors.shape[1]) != store.dimension:
            raise ValueError(
                "Embedding dimension does not match the existing FAISS index."
            )

        store.add(vectors, chunk_batch)
        store.save(index_path)

    return store.count
