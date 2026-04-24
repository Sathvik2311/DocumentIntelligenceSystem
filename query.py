"""CLI entry point for retrieval.

Usage:
    python query.py "what are the key findings?"
    python query.py "..." --top-k 3 --file sample.pdf
    python query.py "..." --doc <doc_id>

Day 4: prints the top-k chunks with similarity scores and citations.
Day 5 will wire this into the LLM to produce grounded answers.
"""

from __future__ import annotations

import argparse
import logging
import sys

from backend.config import get_settings
from backend.services.retrieval import retrieve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the RAG corpus.")
    parser.add_argument("question", help="Natural-language question.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to return.")
    parser.add_argument("--file", dest="filename", default=None, help="Restrict to a single filename.")
    parser.add_argument("--doc", dest="doc_id", default=None, help="Restrict to a single doc_id.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    chunks = retrieve(
        args.question,
        top_k=args.top_k,
        doc_id=args.doc_id,
        filename=args.filename,
    )

    if not chunks:
        print("No results. Ingest a document first with `python ingest.py <file>`.")
        return 1

    print(f"Query: {args.question}\n")
    print(f"Top {len(chunks)} chunks:\n")
    for rank, chunk in enumerate(chunks, start=1):
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"[{rank}] score={chunk.score:.4f}  {chunk.citation}")
        print(f"    {preview}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
