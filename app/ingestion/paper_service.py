"""arXiv identifier normalization and metadata lookup."""

import logging
import re
import threading
import time
from datetime import datetime
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

import httpx

from app.ingestion.errors import MetadataFetchError
from app.ingestion.models import PaperMetadata

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
ARXIV_ID_PATTERN = re.compile(
    r"(?:\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[a-z]{2})?/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"
NAMESPACES = {"atom": ATOM_NAMESPACE, "arxiv": ARXIV_NAMESPACE}
REQUEST_HEADERS = {"User-Agent": "ResearchIQ/0.1 (academic paper ingestion)"}
_ARXIV_API_MIN_INTERVAL_SECONDS = 3.0
_arxiv_api_lock = threading.Lock()
_last_arxiv_api_request_at: float | None = None


def _wait_for_arxiv_api_slot() -> None:
    global _last_arxiv_api_request_at
    with _arxiv_api_lock:
        now = time.monotonic()
        if _last_arxiv_api_request_at is not None:
            delay = _ARXIV_API_MIN_INTERVAL_SECONDS - (now - _last_arxiv_api_request_at)
            if delay > 0:
                time.sleep(delay)
        _last_arxiv_api_request_at = time.monotonic()


def normalize_arxiv_id(value: str) -> str:
    """Return a validated arXiv ID from an ID or arXiv abstract/PDF URL."""
    candidate = value.strip()
    if not candidate:
        raise ValueError("An arXiv ID or URL is required.")

    if "://" in candidate:
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in ARXIV_HOSTS:
            raise ValueError("URL must use an official arXiv host and HTTP or HTTPS.")
        path = unquote(parsed.path).strip("/")
        if path.startswith(("abs/", "pdf/")):
            candidate = path.split("/", maxsplit=1)[1]
        else:
            candidate = path
        if candidate.lower().endswith(".pdf"):
            candidate = candidate[:-4]
    elif candidate.lower().startswith("arxiv:"):
        candidate = candidate[6:]

    match = ARXIV_ID_PATTERN.fullmatch(candidate)
    if match is None:
        raise ValueError(f"Invalid arXiv identifier or URL: {value!r}")
    return match.group(0).lower()


def _required_text(entry: ElementTree.Element, path: str, field_name: str) -> str:
    value = entry.findtext(path, namespaces=NAMESPACES)
    if not value or not value.strip():
        raise MetadataFetchError(f"The arXiv response is missing {field_name}.")
    return " ".join(value.split())


def _parse_datetime(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise MetadataFetchError(f"The arXiv response has an invalid {field_name} date.") from error


def _paper_metadata_from_entry(entry: ElementTree.Element) -> PaperMetadata:
    try:
        entry_id = _required_text(entry, "atom:id", "paper ID")
        canonical_id = normalize_arxiv_id(entry_id)
        authors = [
            " ".join((author.findtext("atom:name", default="", namespaces=NAMESPACES)).split())
            for author in entry.findall("atom:author", NAMESPACES)
        ]
        categories = [
            category.attrib["term"]
            for category in entry.findall("atom:category", NAMESPACES)
            if category.attrib.get("term")
        ]
        pdf_link = next(
            (
                link.attrib["href"]
                for link in entry.findall("atom:link", NAMESPACES)
                if link.attrib.get("title") == "pdf" and link.attrib.get("href")
            ),
            f"https://arxiv.org/pdf/{canonical_id}",
        )
        published = _parse_datetime(_required_text(entry, "atom:published", "published"), "published")
        updated_text = entry.findtext("atom:updated", namespaces=NAMESPACES)
        updated = _parse_datetime(updated_text, "updated") if updated_text else None
        if not authors:
            raise MetadataFetchError("The arXiv response does not contain any authors.")

        return PaperMetadata(
            arxiv_id=canonical_id,
            title=_required_text(entry, "atom:title", "title"),
            authors=authors,
            abstract=_required_text(entry, "atom:summary", "abstract"),
            published=published,
            updated=updated,
            categories=categories,
            pdf_url=pdf_link,
        )
    except (KeyError, ValueError) as error:
        raise MetadataFetchError(f"Could not parse arXiv entry metadata: {error}") from error


def fetch_paper_metadata(arxiv_id_or_url: str) -> PaperMetadata:
    """Fetch and parse a paper's metadata from the official arXiv Atom API."""
    arxiv_id = normalize_arxiv_id(arxiv_id_or_url)
    _wait_for_arxiv_api_slot()
    try:
        response = httpx.get(
            ARXIV_API_URL,
            params={"id_list": arxiv_id, "max_results": 1},
            headers=REQUEST_HEADERS,
            timeout=httpx.Timeout(20.0, connect=10.0),
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise MetadataFetchError(f"Could not fetch arXiv metadata for {arxiv_id}: {error}") from error

    try:
        root = ElementTree.fromstring(response.text)
    except ElementTree.ParseError as error:
        raise MetadataFetchError("The arXiv API returned malformed XML.") from error

    entry = root.find("atom:entry", NAMESPACES)
    if entry is None:
        raise MetadataFetchError(f"arXiv returned no paper for identifier {arxiv_id}.")
    return _paper_metadata_from_entry(entry)


def search_arxiv_papers(query: str, max_results: int = 5) -> list[PaperMetadata]:
    """Search arXiv by query and return a list of parsed metadata objects."""
    if not query or not query.strip():
        raise ValueError("A search query is required.")
    limit = max(1, int(max_results))
    _wait_for_arxiv_api_slot()
    try:
        response = httpx.get(
            ARXIV_API_URL,
            params={"search_query": f"all:{query.strip()}", "start": 0, "max_results": limit},
            headers=REQUEST_HEADERS,
            timeout=httpx.Timeout(25.0, connect=10.0),
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise MetadataFetchError(f"Could not search arXiv for {query!r}: {error}") from error

    try:
        root = ElementTree.fromstring(response.text)
    except ElementTree.ParseError as error:
        raise MetadataFetchError("The arXiv search API returned malformed XML.") from error

    entries = root.findall("atom:entry", NAMESPACES)
    if not entries:
        return []

    papers: list[PaperMetadata] = []
    seen: set[str] = set()
    for entry in entries:
        try:
            metadata = _paper_metadata_from_entry(entry)
        except MetadataFetchError:
            continue
        if metadata.arxiv_id in seen:
            continue
        seen.add(metadata.arxiv_id)
        papers.append(metadata)
    return papers
