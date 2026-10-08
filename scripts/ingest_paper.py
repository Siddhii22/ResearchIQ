"""Run the ResearchIQ ingestion pipeline for one arXiv paper."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.errors import IngestionError
from app.ingestion.pipeline import ingest_paper


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and process an arXiv paper.")
    parser.add_argument("paper", help="arXiv paper ID or arXiv abstract/PDF URL")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        result = ingest_paper(arguments.paper)
    except (IngestionError, ValueError) as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1

    print(f"Paper: {result.metadata.title}")
    print(f"Authors: {', '.join(result.metadata.authors)}")
    print(f"Pages: {len(result.pages)}")
    print(f"Chunks: {len(result.chunks)}")
    print(f"PDF: {result.pdf_path}")
    print(f"Saved to: {result.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())