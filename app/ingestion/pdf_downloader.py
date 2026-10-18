"""Download arXiv PDFs to deterministic local paths."""

import logging
import os
from pathlib import Path
from time import monotonic
from urllib.parse import urljoin, urlsplit

import httpx

from app.ingestion.errors import PDFDownloadError
from app.ingestion.models import PaperMetadata
from app.ingestion.paper_service import REQUEST_HEADERS
from app.ingestion.storage import DEFAULT_PAPERS_DIR, pdf_path_for

logger = logging.getLogger(__name__)
_ARXIV_DOWNLOAD_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
_MAX_PDF_BYTES = 50 * 1024 * 1024
_MAX_PDF_REDIRECTS = 5
_MAX_PDF_DOWNLOAD_SECONDS = 45.0


def _validated_pdf_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise PDFDownloadError("The arXiv response included an invalid PDF URL.") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _ARXIV_DOWNLOAD_HOSTS
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/pdf/")
    ):
        raise PDFDownloadError("The arXiv response included an unsafe PDF URL.")
    return url


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
    download_deadline = monotonic() + _MAX_PDF_DOWNLOAD_SECONDS
    try:
        current_url = _validated_pdf_url(metadata.pdf_url)
        byte_count = 0
        for redirect_count in range(_MAX_PDF_REDIRECTS + 1):
            remaining_seconds = download_deadline - monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError("The arXiv PDF download exceeded its total time limit.")
            with httpx.stream(
                "GET",
                current_url,
                headers=REQUEST_HEADERS,
                timeout=httpx.Timeout(min(10.0, remaining_seconds), connect=min(10.0, remaining_seconds)),
                follow_redirects=False,
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location or redirect_count == _MAX_PDF_REDIRECTS:
                        raise PDFDownloadError("The arXiv PDF redirect chain is invalid or too long.")
                    current_url = _validated_pdf_url(urljoin(current_url, location))
                    continue

                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError as error:
                        raise PDFDownloadError("The arXiv PDF has an invalid content length.") from error
                    if declared_size > _MAX_PDF_BYTES:
                        raise PDFDownloadError("The arXiv PDF exceeds the configured download size limit.")
                with temporary_path.open("wb") as pdf_file:
                    for data in response.iter_bytes():
                        if monotonic() >= download_deadline:
                            raise TimeoutError("The arXiv PDF download exceeded its total time limit.")
                        byte_count += len(data)
                        if byte_count > _MAX_PDF_BYTES:
                            raise PDFDownloadError("The arXiv PDF exceeds the configured download size limit.")
                        pdf_file.write(data)
                break
        else:
            raise PDFDownloadError("The arXiv PDF redirect chain is invalid or too long.")
        if byte_count == 0:
            raise PDFDownloadError(f"arXiv returned an empty PDF for {metadata.arxiv_id}.")
        os.replace(temporary_path, output_path)
    except httpx.HTTPError as error:
        temporary_path.unlink(missing_ok=True)
        raise PDFDownloadError(f"Could not download PDF for {metadata.arxiv_id}: {error}") from error
    except TimeoutError as error:
        temporary_path.unlink(missing_ok=True)
        raise PDFDownloadError(f"PDF download timed out for {metadata.arxiv_id}.") from error
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise PDFDownloadError(f"Could not save PDF for {metadata.arxiv_id}: {error}") from error
    except PDFDownloadError:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.info("Saved PDF to %s", output_path)
    return output_path