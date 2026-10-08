"""Download arXiv PDFs to deterministic local paths."""

import logging
import os
from pathlib import Path

import httpx

from app.ingestion.errors import PDFDownloadError
from app.ingestion.models import PaperMetadata
from app.ingestion.paper_service import REQUEST_HEADERS
from app.ingestion.storage import DEFAULT_PAPERS_DIR, pdf_path_for

logger = logging.getLogger(__name__)


def download_pdf(
    metadata: PaperMetadata,
    data_dir: Path = DEFAULT_PAPERS_DIR,
) -> Path:
    """Download a paper PDF, reusing a non-empty cached file when present."""
    output_path = pdf_path_for(metadata.arxiv_id, data_dir)
    if output_path.is_file() and output_path.stat().st_size > 0:
        logger.info("Using cached PDF at %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".pdf.part")
    logger.info("Downloading arXiv PDF %s", metadata.arxiv_id)
    try:
        with httpx.stream(
            "GET",
            metadata.pdf_url,
            headers=REQUEST_HEADERS,
            timeout=httpx.Timeout(45.0, connect=10.0),
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            byte_count = 0
            with temporary_path.open("wb") as pdf_file:
                for data in response.iter_bytes():
                    pdf_file.write(data)
                    byte_count += len(data)
        if byte_count == 0:
            raise PDFDownloadError(f"arXiv returned an empty PDF for {metadata.arxiv_id}.")
        os.replace(temporary_path, output_path)
    except httpx.HTTPError as error:
        temporary_path.unlink(missing_ok=True)
        raise PDFDownloadError(f"Could not download PDF for {metadata.arxiv_id}: {error}") from error
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise PDFDownloadError(f"Could not save PDF for {metadata.arxiv_id}: {error}") from error
    except PDFDownloadError:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.info("Saved PDF to %s", output_path)
    return output_path