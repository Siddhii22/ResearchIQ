"""End-to-end local paper ingestion orchestration."""

import logging
from pathlib import Path

from app.ingestion.chunking import chunk_paper
from app.ingestion.models import IngestionResult
from app.ingestion.paper_service import fetch_paper_metadata
from app.ingestion.pdf_downloader import download_pdf
from app.ingestion.pdf_extractor import extract_pdf_pages
from app.ingestion.storage import DEFAULT_PAPERS_DIR, save_processed_paper

logger = logging.getLogger(__name__)


def ingest_paper(
    arxiv_id_or_url: str,
    data_dir: Path = DEFAULT_PAPERS_DIR,
) -> IngestionResult:
    """Fetch metadata, cache a PDF, extract pages, chunk, and save JSON."""
    metadata = fetch_paper_metadata(arxiv_id_or_url)
    pdf_path = download_pdf(metadata, data_dir)
    pages = extract_pdf_pages(pdf_path)
    chunks = chunk_paper(metadata, pages)
    json_path = save_processed_paper(metadata, pages, chunks, data_dir)
    logger.info("Ingested %s into %d chunks", metadata.arxiv_id, len(chunks))

    return IngestionResult(
        metadata=metadata,
        pages=pages,
        chunks=chunks,
        pdf_path=str(pdf_path),
        json_path=str(json_path),
    )