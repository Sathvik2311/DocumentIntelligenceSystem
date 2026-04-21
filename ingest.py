"""CLI entry point for the ingestion pipeline.

Usage:
    python ingest.py <file_path>

Day 2: prints parse/chunk summary.
Day 3 will extend this to embed + persist to ChromaDB.
"""

from __future__ import annotations

import argparse
import logging
import sys

from backend.config import get_settings
from backend.services.ingestion import ingest_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a document.")
    parser.add_argument("path", help="Path to a PDF, DOCX, or TXT file.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s")

    chunks = ingest_file(args.path)
    if not chunks:
        print(f"No chunks produced from {args.path} (empty or unreadable).")
        return 1

    first = chunks[0]
    print(f"File:       {first.filename}")
    print(f"Doc ID:     {first.doc_id}")
    print(f"Chunks:     {len(chunks)}")
    print(f"Pages seen: {len({c.page_number for c in chunks})}")
    print(f"Tokens/chk: min={min(c.token_count for c in chunks)} "
          f"max={max(c.token_count for c in chunks)}")
    print(f"\n--- First chunk (page {first.page_number}, idx {first.chunk_index}) ---")
    print(first.text[:400] + ("..." if len(first.text) > 400 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
