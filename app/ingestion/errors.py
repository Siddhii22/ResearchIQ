"""Exceptions raised by paper ingestion operations."""


class IngestionError(Exception):
    """Base error for a failed paper ingestion operation."""


class MetadataFetchError(IngestionError):
    """The arXiv API could not provide usable paper metadata."""


class PDFDownloadError(IngestionError):
    """The paper PDF could not be downloaded or saved."""


class PDFExtractionError(IngestionError):
    """The downloaded PDF could not be read as extractable text."""