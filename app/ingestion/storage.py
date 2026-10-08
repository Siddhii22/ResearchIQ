"""Deterministic local paths and JSON storage for ingested papers."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from app.ingestion.models import ExtractedPage, PaperChunk, PaperMetadata
from app.ingestion.paper_service import normalize_arxiv_id

logger = logging.getLogger(__name__)
DEFAULT_PAPERS_DIR = Path("data/papers")


def paper_storage_stem(arxiv_id: str) -> str:
    """Make a filesystem-safe, deterministic stem for an arXiv identifier."""
    return normalize_arxiv_id(arxiv_id).replace("/", "_")


def pdf_path_for(arxiv_id: str, data_dir: Path = DEFAULT_PAPERS_DIR) -> Path:
    """Return the deterministic local PDF path for an arXiv paper."""
    return data_dir / f"{paper_storage_stem(arxiv_id)}.pdf"


def processed_path_for(arxiv_id: str, data_dir: Path = DEFAULT_PAPERS_DIR) -> Path:
    """Return the deterministic processed JSON path for an arXiv paper."""
    return data_dir / f"{paper_storage_stem(arxiv_id)}.json"


def save_processed_paper(
    metadata: PaperMetadata,
    pages: list[ExtractedPage],
    chunks: list[PaperChunk],
    data_dir: Path = DEFAULT_PAPERS_DIR,
) -> Path:
    """Serialize metadata and page-linked chunks as one JSON document."""
    output_path = processed_path_for(metadata.arxiv_id, data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paper_data = asdict(metadata)
    paper_data["published"] = metadata.published.isoformat()
    paper_data["updated"] = metadata.updated.isoformat() if metadata.updated else None
    document = {
        "paper": paper_data,
        "page_count": len(pages),
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    temporary_path = output_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.info("Saved processed paper data to %s", output_path)
    return output_path