"""Page-preserving, section-aware paper chunking."""

import re

from app.ingestion.models import ExtractedPage, PaperChunk, PaperMetadata

MAX_CHUNK_CHARACTERS = 1800
SECTION_NAMES = (
    "Abstract",
    "Introduction",
    "Related Work",
    "Background",
    "Methods",
    "Methodology",
    "Experiments",
    "Experimental Setup",
    "Results",
    "Discussion",
    "Conclusion",
    "References",
)
_SECTION_PATTERN = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*|[A-Z])[.)]?\s+)?("
    + "|".join(re.escape(name) for name in sorted(SECTION_NAMES, key=len, reverse=True))
    + r")\s*:?[.]?\s*$",
    re.IGNORECASE,
)


def _section_heading(line: str) -> str | None:
    match = _SECTION_PATTERN.fullmatch(line)
    if match is None:
        return None
    return next(name for name in SECTION_NAMES if name.lower() == match.group(1).lower())


def _split_long_paragraph(paragraph: str, limit: int) -> list[str]:
    """Split only an overlong paragraph, preferring sentence and word boundaries."""
    if len(paragraph) <= limit:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                pieces.append(current)
                current = ""
            words = sentence.split()
            word_chunk = ""
            for word in words:
                if len(word) > limit:
                    if word_chunk:
                        pieces.append(word_chunk)
                        word_chunk = ""
                    pieces.extend(word[index : index + limit] for index in range(0, len(word), limit))
                elif not word_chunk:
                    word_chunk = word
                elif len(word_chunk) + len(word) + 1 <= limit:
                    word_chunk = f"{word_chunk} {word}"
                else:
                    pieces.append(word_chunk)
                    word_chunk = word
            if word_chunk:
                current = word_chunk
        elif not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_paper(
    metadata: PaperMetadata,
    pages: list[ExtractedPage],
    max_characters: int = MAX_CHUNK_CHARACTERS,
) -> list[PaperChunk]:
    """Create chunks grouped by detected section and page, falling back to paragraphs."""
    if max_characters < 1:
        raise ValueError("max_characters must be a positive integer.")

    chunks: list[PaperChunk] = []
    current_section = "Unsectioned"

    def add_chunk(text: str, section: str, page_number: int) -> None:
        if not text.strip():
            return
        chunk_number = len(chunks) + 1
        chunks.append(
            PaperChunk(
                chunk_id=f"{metadata.arxiv_id.replace('/', '_')}-chunk-{chunk_number:04d}",
                text=text.strip(),
                paper_id=metadata.arxiv_id,
                title=metadata.title,
                authors=metadata.authors,
                section=section,
                page=page_number,
            )
        )

    for page in pages:
        segments: list[tuple[str, str]] = []
        paragraph_lines: list[str] = []

        def flush_paragraph() -> None:
            if paragraph_lines:
                paragraph = " ".join(line.strip() for line in paragraph_lines).strip()
                if paragraph:
                    segments.append((current_section, paragraph))
                paragraph_lines.clear()

        for line in page.text.splitlines():
            heading = _section_heading(line)
            if heading is not None:
                flush_paragraph()
                current_section = heading
            elif line.strip():
                paragraph_lines.append(line)
            else:
                flush_paragraph()
        flush_paragraph()

        active_section: str | None = None
        active_text = ""
        for section, paragraph in segments:
            for piece in _split_long_paragraph(paragraph, max_characters):
                if section != active_section:
                    add_chunk(active_text, active_section or "Unsectioned", page.page_number)
                    active_section = section
                    active_text = piece
                elif not active_text:
                    active_text = piece
                elif len(active_text) + len(piece) + 1 <= max_characters:
                    active_text = f"{active_text} {piece}"
                else:
                    add_chunk(active_text, section, page.page_number)
                    active_text = piece
        if active_section is not None:
            add_chunk(active_text, active_section, page.page_number)

    return chunks