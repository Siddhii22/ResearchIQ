"""Offline tests for arXiv ingestion helpers."""

from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from app.ingestion.chunking import chunk_paper
from app.ingestion.errors import PDFDownloadError
from app.ingestion.models import ExtractedPage, PaperMetadata
from app.ingestion.paper_service import fetch_paper_metadata, normalize_arxiv_id
from app.ingestion.pdf_downloader import download_pdf
from app.ingestion.storage import pdf_path_for, processed_path_for

ATOM_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-03T10:00:00Z</updated>
    <published>2024-01-02T09:00:00Z</published>
    <title>  Evidence Based Research  </title>
    <summary>A short abstract for testing.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="https://arxiv.org/pdf/2401.12345" rel="related" title="pdf" />
    <arxiv:primary_category term="cs.CL" />
    <category term="cs.CL" />
    <category term="cs.IR" />
  </entry>
</feed>
"""


def sample_metadata() -> PaperMetadata:
    return PaperMetadata(
        arxiv_id="2401.12345",
        title="Evidence Based Research",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="A short abstract for testing.",
        published=datetime(2024, 1, 2, tzinfo=timezone.utc),
        updated=datetime(2024, 1, 3, tzinfo=timezone.utc),
        categories=["cs.CL", "cs.IR"],
        pdf_url="https://arxiv.org/pdf/2401.12345",
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2401.12345", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345v2", "2401.12345v2"),
        ("https://arxiv.org/pdf/hep-th/9901001.pdf", "hep-th/9901001"),
        ("arXiv:2401.12345", "2401.12345"),
    ],
)
def test_normalize_arxiv_id(value: str, expected: str) -> None:
    assert normalize_arxiv_id(value) == expected


def test_normalize_rejects_non_arxiv_urls() -> None:
    with pytest.raises(ValueError, match="official arXiv host"):
        normalize_arxiv_id("https://example.com/abs/2401.12345")


def test_metadata_lookup_parses_atom_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        assert url == "https://export.arxiv.org/api/query"
        assert kwargs["params"] == {"id_list": "2401.12345", "max_results": 1}
        return httpx.Response(200, text=ATOM_RESPONSE, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    metadata = fetch_paper_metadata("2401.12345")

    assert metadata.arxiv_id == "2401.12345v1"
    assert metadata.title == "Evidence Based Research"
    assert metadata.authors == ["Ada Lovelace", "Alan Turing"]
    assert metadata.abstract == "A short abstract for testing."
    assert metadata.published == datetime(2024, 1, 2, 9, tzinfo=timezone.utc)
    assert metadata.categories == ["cs.CL", "cs.IR"]
    assert metadata.pdf_url == "https://arxiv.org/pdf/2401.12345"


def test_metadata_model_is_typed() -> None:
    metadata = sample_metadata()

    assert metadata.arxiv_id == "2401.12345"
    assert isinstance(metadata.published, datetime)
    assert metadata.updated is not None
    assert metadata.authors == ["Ada Lovelace", "Alan Turing"]


def test_paper_filenames_are_deterministic(tmp_path: Path) -> None:
    assert pdf_path_for("2401.12345", tmp_path) == tmp_path / "2401.12345.pdf"
    assert pdf_path_for("hep-th/9901001", tmp_path) == tmp_path / "hep-th_9901001.pdf"
    assert processed_path_for("2401.12345", tmp_path) == tmp_path / "2401.12345.json"


def test_pdf_download_reuses_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached_path = pdf_path_for("2401.12345", tmp_path)
    cached_path.write_bytes(b"cached PDF")

    def unexpected_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("A cached PDF should not be downloaded again.")

    monkeypatch.setattr(httpx, "stream", unexpected_download)

    assert download_pdf(sample_metadata(), tmp_path) == cached_path
    assert cached_path.read_bytes() == b"cached PDF"


def test_pdf_download_reports_http_failure_and_cleans_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def failing_stream(*args: object, **kwargs: object) -> Iterator[httpx.Response]:
        request = httpx.Request("GET", "https://arxiv.org/pdf/2401.12345")
        yield httpx.Response(503, request=request)

    monkeypatch.setattr(httpx, "stream", failing_stream)

    with pytest.raises(PDFDownloadError, match="Could not download PDF"):
        download_pdf(sample_metadata(), tmp_path)

    assert not pdf_path_for("2401.12345", tmp_path).exists()
    assert not list(tmp_path.glob("*.part"))


def test_chunking_preserves_detected_sections_and_page_metadata() -> None:
    metadata = sample_metadata()
    pages = [
        ExtractedPage(
            page_number=3,
            text=(
                "Abstract\nThis paper studies evidence retrieval.\n\n"
                "1 Introduction\nPrior work motivates our problem.\n\n"
                "Methods\nWe compare two approaches.\n\n"
                "References\n[1] Example reference."
            ),
        )
    ]

    chunks = chunk_paper(metadata, pages, max_characters=80)

    assert [chunk.section for chunk in chunks] == [
        "Abstract",
        "Introduction",
        "Methods",
        "References",
    ]
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.paper_id == metadata.arxiv_id for chunk in chunks)
    assert all(chunk.title == metadata.title for chunk in chunks)
    assert all(chunk.authors == metadata.authors and chunk.text for chunk in chunks)


def test_chunking_falls_back_to_page_paragraphs() -> None:
    chunks = chunk_paper(
        sample_metadata(),
        [ExtractedPage(page_number=1, text="A useful first paragraph.\n\nA second paragraph.")],
    )

    assert len(chunks) == 1
    assert chunks[0].section == "Unsectioned"
    assert chunks[0].page == 1
    assert "A useful first paragraph." in chunks[0].text
    assert "A second paragraph." in chunks[0].text


def test_long_paragraph_is_split_at_sentence_boundaries() -> None:
    chunks = chunk_paper(
        sample_metadata(),
        [ExtractedPage(page_number=2, text="First sentence. Second sentence. Third sentence.")],
        max_characters=25,
    )

    assert len(chunks) == 3
    assert all(len(chunk.text) <= 25 for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks) == (
        "First sentence. Second sentence. Third sentence."
    )