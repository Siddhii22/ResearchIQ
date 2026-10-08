"""Extract page-preserving text from research paper PDFs."""

import logging
from pathlib import Path

from pypdf import PdfReader

from app.ingestion.errors import PDFExtractionError
from app.ingestion.models import ExtractedPage

logger = logging.getLogger(__name__)


def extract_pdf_pages(pdf_path: Path) -> list[ExtractedPage]:
    """Extract text from each PDF page without attempting OCR."""
    if not pdf_path.is_file():
        raise PDFExtractionError(f"PDF file does not exist: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path), strict=False)
        if reader.is_encrypted:
            raise PDFExtractionError(f"PDF is encrypted and cannot be read: {pdf_path}")

        pages = [
            ExtractedPage(page_number=index, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages, start=1)
        ]
    except PDFExtractionError:
        raise
    except Exception as error:
        raise PDFExtractionError(f"Could not parse PDF {pdf_path}: {error}") from error

    if not pages:
        raise PDFExtractionError(f"PDF contains no pages: {pdf_path}")
    if not any(page.text.strip() for page in pages):
        raise PDFExtractionError(
            f"No extractable text was found in {pdf_path}; scanned PDFs need OCR, which is not supported."
        )

    logger.info("Extracted text from %d pages in %s", len(pages), pdf_path)
    return pages