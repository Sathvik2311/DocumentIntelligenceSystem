"""CLI entry point for the ingestion pipeline.

Usage:
    python ingest.py <file_path>

Parses the file, chunks it, embeds each chunk with sentence-transformers,
and upserts the chunks into the ChromaDB collection at CHROMA_PERSIST_DIR.
"""

from __future__ import annotations

import argparse
import logging
import sys

from backend.config import get_settings
from backend.services.ingestion import get_collection, ingest_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest a document into ChromaDB.")
    parser.add_argument("path", help="Path to a PDF, DOCX, or TXT file.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = ingest_document(args.path)
    collection_total = get_collection().count()

    print(f"File:              {result.filename}")
    print(f"Doc ID:            {result.doc_id}")
    print(f"Pages parsed:      {result.num_pages}")
    print(f"Chunks stored:     {result.num_chunks}")
    print(f"Collection total:  {collection_total}")
    return 0 if result.num_chunks > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
