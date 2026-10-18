"""API validation and response tests with a mocked RAG pipeline."""

from collections.abc import Iterator
import logging
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.limits import MAX_RESEARCH_BODY_BYTES, PerClientRateLimiter
from app.api.research import get_rag_pipeline, refresh_relevant_papers
from app.ingestion.errors import MetadataFetchError
from app.main import app
from app.services.llm_service import LLMProviderUnavailableError, LLMQuotaError, LLMTimeoutError
from app.services.rag_pipeline import EvidenceSource, RAGAnswer


class FakePipeline:
    def answer_question(self, question: str, top_k: int = 5) -> RAGAnswer:
        assert question == "What does the Transformer use?"
        assert top_k == 1
        return RAGAnswer(
            answer="It relies on attention [1].",
            sources=[
                EvidenceSource(
                    citation_id=1,
                    chunk_id="retrieved-chunk",
                    paper_id="1706.03762v7",
                    title="Attention Is All You Need",
                    authors=["Ashish Vaswani"],
                    section="Introduction",
                    page=3,
                    text="The Transformer relies on attention.",
                    similarity_score=0.9,
                    retrieval_score=4.25,
                    retrieval_method="hybrid_rrf_reranked",
                )
            ],
        )


class TwoPaperPipeline:
    def answer_question(self, question: str, top_k: int = 5) -> RAGAnswer:
        return RAGAnswer(
            answer="The first claim is supported by [1], and the second by [2].",
            sources=[
                EvidenceSource(
                    citation_id=1,
                    chunk_id="paper-one-chunk-0001",
                    paper_id="1706.03762v7",
                    title="Attention Is All You Need",
                    authors=["Ashish Vaswani"],
                    section="Introduction",
                    page=3,
                    text="Evidence from the first paper.",
                    similarity_score=0.9,
                ),
                EvidenceSource(
                    citation_id=2,
                    chunk_id="paper-two-chunk-0001",
                    paper_id="2401.12345v1",
                    title="A Different Research Paper",
                    authors=["Ada Lovelace"],
                    section="Methods",
                    page=7,
                    text="Evidence from the second paper.",
                    similarity_score=0.8,
                ),
            ][:top_k],
        )


class UnavailablePipeline:
    def answer_question(self, question: str, top_k: int = 5) -> RAGAnswer:
        raise LLMProviderUnavailableError(
            "Gemini is temporarily unavailable. Please try again shortly."
        )


class TimedOutPipeline:
    def answer_question(self, question: str, top_k: int = 5) -> RAGAnswer:
        raise LLMTimeoutError("The Gemini request timed out.")


class QuotaPipeline:
    def answer_question(self, question: str, top_k: int = 5) -> RAGAnswer:
        raise LLMQuotaError("Gemini quota or rate limit exceeded. Check the project's quota and billing.")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    pipeline = FakePipeline()
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    monkeypatch.setattr("app.api.research.refresh_relevant_papers", lambda *args, **kwargs: 0)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_research_query_returns_answer_and_retrieved_sources(client: TestClient) -> None:
    response = client.post(
        "/api/v1/research/query",
        json={"question": "What does the Transformer use?", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "It relies on attention [1]."
    assert response.json()["sources"][0] == {
        "citation_id": "[1]",
        "chunk_id": "retrieved-chunk",
        "paper_id": "1706.03762v7",
        "paper_url": "https://arxiv.org/abs/1706.03762v7",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani"],
        "section": "Introduction",
        "page": 3,
        "text": "The Transformer relies on attention.",
        "similarity_score": 0.9,
        "retrieval_score": 4.25,
        "retrieval_method": "hybrid_rrf_reranked",
    }


def test_research_api_preserves_distinct_source_titles_ids_and_citations(
    client: TestClient,
) -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: TwoPaperPipeline()

    response = client.post(
        "/api/v1/research/query",
        json={"question": "What does the Transformer use?", "top_k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The first claim is supported by [1], and the second by [2]."
    assert [
        (source["citation_id"], source["chunk_id"], source["paper_id"], source["title"])
        for source in body["sources"]
    ] == [
        ("[1]", "paper-one-chunk-0001", "1706.03762v7", "Attention Is All You Need"),
        ("[2]", "paper-two-chunk-0001", "2401.12345v1", "A Different Research Paper"),
    ]


def test_research_query_maps_gemini_unavailable_to_503(client: TestClient) -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: UnavailablePipeline()

    response = client.post(
        "/api/v1/research/query",
        json={"question": "What does the Transformer use?", "top_k": 1},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Gemini is temporarily unavailable. Please try again shortly."
    }


def test_research_query_maps_llm_timeout_to_504(client: TestClient) -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: TimedOutPipeline()

    response = client.post(
        "/api/v1/research/query",
        json={"question": "What does the Transformer use?", "top_k": 1},
    )

    assert response.status_code == 504
    assert response.json() == {"detail": "The Gemini request timed out."}


def test_research_query_maps_gemini_quota_to_429(client: TestClient) -> None:
    app.dependency_overrides[get_rag_pipeline] = lambda: QuotaPipeline()

    response = client.post(
        "/api/v1/research/query",
        json={"question": "What does the Transformer use?", "top_k": 1},
    )

    assert response.status_code == 429
    assert response.json() == {
        "detail": "Gemini quota or rate limit exceeded. Check the project's quota and billing."
    }


def test_refresh_relevant_papers_rebuilds_index_for_new_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sample = {
        "arxiv_id": "2401.30000v1",
        "title": "New Retrieval Study",
        "authors": ["Ada Lovelace"],
        "abstract": "A study about retrieval.",
        "published": "2024-01-02T09:00:00Z",
        "updated": "2024-01-03T09:00:00Z",
        "categories": ["cs.IR"],
        "pdf_url": "https://arxiv.org/pdf/2401.30000v1.pdf",
    }
    result = type(
        "DummyResult",
        (),
        {"chunks": [type("Chunk", (), {"text": "n/a"})()]},
    )()

    monkeypatch.setattr(
        "app.api.research.search_arxiv_papers",
        lambda question, max_results=5: [type("Paper", (), sample)()],
    )
    def fake_ingest(arxiv_id_or_url, data_dir, metadata):
        assert metadata.arxiv_id == "2401.30000v1"
        return result

    monkeypatch.setattr("app.api.research.ingest_paper", fake_ingest)
    monkeypatch.setattr("app.api.research.append_processed_chunks", lambda papers_dir, index_path: 1)
    monkeypatch.setattr("app.api.research.get_rag_pipeline.cache_clear", lambda: None)

    indexed_count = refresh_relevant_papers(
        "new retrieval study",
        papers_dir=tmp_path / "papers",
        index_path=tmp_path / "index.faiss",
        max_results=3,
    )

    assert indexed_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "ok"},
        {"question": "A valid question", "top_k": 0},
    ],
)
def test_research_query_validates_request(payload: dict[str, object], client: TestClient) -> None:
    response = client.post("/api/v1/research/query", json=payload)

    assert response.status_code == 422


def test_research_query_rejects_oversized_body(client: TestClient) -> None:
    question = b"a" * MAX_RESEARCH_BODY_BYTES
    response = client.post(
        "/api/v1/research/query",
        content=b'{"question":"' + question + b'"}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the allowed size."}


def test_research_rate_limiter_rejects_excess_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr("app.api.limits.monotonic", lambda: now[0])
    limiter = PerClientRateLimiter(requests=2, window_seconds=60)

    limiter.check("client-a")
    limiter.check("client-a")
    with pytest.raises(HTTPException) as error:
        limiter.check("client-a")

    assert error.value.status_code == 429
    now[0] = 161.0
    limiter.check("client-a")


def test_arxiv_failure_logs_do_not_include_research_question(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_question = "private-question-marker"

    def failed_search(query: str, max_results: int = 5):
        raise MetadataFetchError(f"Search failed for {query}")

    monkeypatch.setattr("app.api.research.search_arxiv_papers", failed_search)
    caplog.set_level(logging.WARNING, logger="app.api.research")

    assert refresh_relevant_papers(private_question) == 0
    assert private_question not in caplog.text


def test_index_failure_response_does_not_reveal_local_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    private_path = str(tmp_path / "private-index-location")
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "paper.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("app.api.research.DEFAULT_INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr("app.api.research.DEFAULT_PAPERS_DIR", papers_dir)
    monkeypatch.setattr(
        "app.api.research.build_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError(private_path)),
    )
    get_rag_pipeline.cache_clear()

    with pytest.raises(HTTPException) as error:
        get_rag_pipeline()

    assert error.value.status_code == 503
    assert private_path not in str(error.value.detail)