"""Typed data models used by the paper ingestion pipeline."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PaperMetadata:
    """Metadata returned by the arXiv Atom API."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime | None
    categories: list[str]
    pdf_url: str


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from one PDF page, numbered from one."""

    page_number: int
    text: str


@dataclass(frozen=True)
class PaperChunk:
    """A page-local text chunk with source paper metadata."""

    chunk_id: str
    text: str
    paper_id: str
    title: str
    authors: list[str]
    section: str
    page: int


@dataclass(frozen=True)
class IngestionResult:
    """Files and parsed content produced by a completed ingestion run."""

    metadata: PaperMetadata
    pages: list[ExtractedPage]
    chunks: list[PaperChunk]
    pdf_path: str
    json_path: str