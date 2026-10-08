"""Offline tests for BM25, RRF, reranking, and hybrid metadata retention."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from app.retrieval.bm25_retriever import BM25Retriever, tokenize
from app.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import RetrievedChunk
from app.retrieval.vector_store import FaissVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    page: int = 1,
    section: str = "Introduction",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id="1706.03762v7",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        section=section,
        page=page,
        text=text,
        similarity_score=0.8,
    )


class FakeDenseRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.requested_top_k: int | None = None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.requested_top_k = top_k
        return self.chunks[:top_k]


class FakeReranker:
    def __init__(self, scores: list[float] | dict[str, float]) -> None:
        self.scores = scores
        self.received: list[RetrievedChunk] = []

    def score(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        self.received = list(chunks)
        if isinstance(self.scores, dict):
            return [self.scores[chunk.chunk_id] for chunk in chunks]
        return self.scores[: len(chunks)]


def as_metadata(chunk: RetrievedChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "paper_id": chunk.paper_id,
        "title": chunk.title,
        "authors": chunk.authors,
        "section": chunk.section,
        "page": chunk.page,
        "text": chunk.text,
        "citation_id": f"cite-{chunk.chunk_id}",
    }


def test_tokenization_is_lowercase_and_deterministic() -> None:
    expected = ["self", "attention", "q", "k", "v", "2024"]
    assert tokenize("Self-Attention: Q/K/V, 2024!") == expected
    assert tokenize("Self-Attention: Q/K/V, 2024!") == expected


def test_bm25_retrieves_relevant_chunk_and_preserves_metadata() -> None:
    chunks = [
        as_metadata(make_chunk("unrelated", "Translation scores improve on the benchmark.", page=9)),
        as_metadata(
            make_chunk(
                "self-attention",
                "The Transformer uses self-attention in encoder and decoder layers.",
                page=4,
                section="Methods",
            )
        ),
    ]
    results = BM25Retriever(chunks).retrieve("self-attention encoder", top_k=2)

    assert results[0].chunk_id == "self-attention"
    assert results[0].retrieval_method == "bm25"
    assert results[0].retrieval_score is not None and results[0].retrieval_score > 0
    assert results[0].paper_id == "1706.03762v7"
    assert results[0].title == "Attention Is All You Need"
    assert results[0].authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert results[0].section == "Methods"
    assert results[0].page == 4
    assert results[0].text.startswith("The Transformer")


def test_bm25_ties_and_repeated_calls_are_deterministic() -> None:
    chunks = [
        as_metadata(make_chunk(chunk_id, "same words here", page=page))
        for chunk_id, page in [("first", 1), ("second", 2)]
    ]
    retriever = BM25Retriever(chunks)

    first = retriever.retrieve("same words", top_k=2)
    second = retriever.retrieve("same words", top_k=2)

    assert [chunk.chunk_id for chunk in first] == ["first", "second"]
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.retrieval_score for chunk in first] == [chunk.retrieval_score for chunk in second]


def test_rrf_uses_one_based_ranks_and_deduplicates_by_chunk_id() -> None:
    first = make_chunk("shared", "Dense first")
    dense_second = make_chunk("dense-second", "Dense second")
    lexical_first = make_chunk("shared", "BM25 copy")
    lexical_second = make_chunk("lexical-second", "BM25 second")

    fused = reciprocal_rank_fusion(
        [[first, dense_second], [lexical_first, lexical_second]],
        k=10,
    )

    assert [candidate.chunk.chunk_id for candidate in fused] == [
        "shared",
        "dense-second",
        "lexical-second",
    ]
    assert fused[0].rrf_score == pytest.approx(1 / 11 + 1 / 11)
    assert fused[1].rrf_score == pytest.approx(1 / 12)
    assert fused[2].rrf_score == pytest.approx(1 / 12)
    assert fused[0].chunk.text == "Dense first"


def test_rrf_ties_preserve_first_seen_order_and_validate_k() -> None:
    fused = reciprocal_rank_fusion(
        [[make_chunk("first", "x"), make_chunk("second", "x")]],
        k=60,
    )

    assert [candidate.chunk.chunk_id for candidate in fused] == ["first", "second"]
    with pytest.raises(ValueError, match="at least 1"):
        reciprocal_rank_fusion([], k=0)


def test_hybrid_pipeline_limits_candidates_reranks_and_preserves_sources() -> None:
    chunks = [
        make_chunk("dense-top", "Transformer architecture uses self-attention", page=3),
        make_chunk(
            "lexical-top",
            "Self-attention computes query key value products",
            page=4,
            section="Methods",
        ),
        make_chunk("other", "Translation results and evaluation", page=8, section="Results"),
    ]
    dense = FakeDenseRetriever([chunks[0], chunks[2]])
    bm25 = BM25Retriever([as_metadata(chunk) for chunk in chunks])
    reranker = FakeReranker({"dense-top": 0.1, "lexical-top": 0.9, "other": 0.4})
    retriever = HybridRetriever(
        dense,
        bm25,
        reranker,
        dense_top_k=2,
        bm25_top_k=2,
        rrf_k=60,
        rerank_top_k=3,
        final_top_k=2,
    )

    results = retriever.retrieve("self-attention Transformer", top_k=2)

    assert dense.requested_top_k == 2
    assert len(reranker.received) == 3
    assert [chunk.chunk_id for chunk in results] == ["lexical-top", "other"]
    assert results[0].retrieval_method == "hybrid_rrf_reranked"
    assert results[0].retrieval_score == 0.9
    assert results[0].paper_id == "1706.03762v7"
    assert results[0].title == "Attention Is All You Need"
    assert results[0].authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert results[0].section == "Methods"
    assert results[0].page == 4
    assert results[0].text == chunks[1].text


def test_cross_encoder_model_load_is_lazy_and_cpu_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
            assert kwargs["show_progress_bar"] is False
            return [0.7 for _ in pairs]

    factory = Mock(return_value=FakeCrossEncoder())
    fake_sentence_transformers = ModuleType("sentence_transformers")
    fake_sentence_transformers.CrossEncoder = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    from app.retrieval.reranker import _load_cross_encoder

    _load_cross_encoder.cache_clear()
    reranker = CrossEncoderReranker("test-cross-encoder")
    factory.assert_not_called()

    scores = reranker.score("query", [make_chunk("one", "text")])

    assert scores == [0.7]
    factory.assert_called_once_with("test-cross-encoder", device="cpu")
    _load_cross_encoder.cache_clear()


def test_vector_store_exposes_copied_chunk_metadata() -> None:
    store = FaissVectorStore(dimension=2)
    chunk_metadata = as_metadata(make_chunk("source-one", "Some evidence.", page=2, section="Abstract"))
    store.add([[1.0, 0.0]], [chunk_metadata])

    exported = store.chunks
    exported[0]["title"] = "mutated copy"

    assert store.chunks[0]["title"] == "Attention Is All You Need"
    assert store.chunks[0]["citation_id"] == "cite-source-one"


def test_hybrid_candidate_limits_are_configurable() -> None:
    dense = FakeDenseRetriever([make_chunk("dense", "self attention")])
    bm25 = BM25Retriever([as_metadata(make_chunk("bm25", "self attention"))])
    reranker = FakeReranker([0.5, 0.4])
    retriever = HybridRetriever(
        dense,
        bm25,
        reranker,
        dense_top_k=1,
        bm25_top_k=1,
        rrf_k=7,
        rerank_top_k=1,
        final_top_k=1,
    )

    result = retriever.retrieve("self attention")

    assert len(reranker.received) == 1
    assert len(result) == 1