"""End-to-end local paper ingestion orchestration."""

import logging
from pathlib import Path
from time import perf_counter

from app.ingestion.chunking import chunk_paper
from app.ingestion.models import IngestionResult, PaperMetadata
from app.ingestion.paper_service import fetch_paper_metadata
from app.ingestion.pdf_downloader import download_pdf
from app.ingestion.pdf_extractor import extract_pdf_pages
from app.ingestion.storage import DEFAULT_PAPERS_DIR, save_processed_paper

logger = logging.getLogger(__name__)


def _run_timed_stage(stage: str, paper_id: str, operation):
    started = perf_counter()
    logger.info("ingestion_stage_started stage=%s paper_id=%s", stage, paper_id)
    try:
        result = operation()
    except Exception as error:
        logger.warning(
            "ingestion_stage_failed stage=%s paper_id=%s elapsed_seconds=%.3f error_type=%s",
            stage,
            paper_id,
            perf_counter() - started,
            type(error).__name__,
        )
        raise
    logger.info(
        "ingestion_stage_completed stage=%s paper_id=%s elapsed_seconds=%.3f",
        stage,
        paper_id,
        perf_counter() - started,
    )
    return result


def ingest_paper(
    arxiv_id_or_url: str,
    data_dir: Path = DEFAULT_PAPERS_DIR,
    metadata: PaperMetadata | None = None,
) -> IngestionResult:
    """Fetch metadata, cache a PDF, extract pages, chunk, and save JSON."""
    metadata = metadata or fetch_paper_metadata(arxiv_id_or_url)
    pdf_path = _run_timed_stage(
        "pdf_download",
        metadata.arxiv_id,
        lambda: download_pdf(metadata, data_dir),
    )
    pages = _run_timed_stage(
        "pdf_extraction",
        metadata.arxiv_id,
        lambda: extract_pdf_pages(pdf_path),
    )
    chunks = _run_timed_stage(
        "chunking",
        metadata.arxiv_id,
        lambda: chunk_paper(metadata, pages),
    )
    json_path = _run_timed_stage(
        "paper_cache_write",
        metadata.arxiv_id,
        lambda: save_processed_paper(metadata, pages, chunks, data_dir),
    )
    logger.info("Ingested %s into %d chunks", metadata.arxiv_id, len(chunks))

    return IngestionResult(
        metadata=metadata,
        pages=pages,
        chunks=chunks,
        pdf_path=str(pdf_path),
        json_path=str(json_path),
    )