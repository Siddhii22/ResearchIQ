"""API validation and response tests with a mocked RAG pipeline."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.research import get_rag_pipeline
from app.main import app
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


@pytest.fixture
def client() -> Iterator[TestClient]:
    pipeline = FakePipeline()
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
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
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani"],
        "section": "Introduction",
        "page": 3,
        "text": "The Transformer relies on attention.",
        "similarity_score": 0.9,
        "retrieval_score": 4.25,
        "retrieval_method": "hybrid_rrf_reranked",
    }


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