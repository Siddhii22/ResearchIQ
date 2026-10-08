"""Build the local FAISS index from Phase 2 processed papers."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.index_builder import build_index
from app.retrieval.vector_store import DEFAULT_INDEX_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed processed papers and build a FAISS index.")
    parser.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    arguments = parser.parse_args()

    try:
        count = build_index(arguments.papers_dir, arguments.index_path)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Index build failed: {error}", file=sys.stderr)
        return 1

    print(f"Indexed chunks: {count}")
    print(f"FAISS index: {arguments.index_path}")
    print(f"Metadata: {arguments.index_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())