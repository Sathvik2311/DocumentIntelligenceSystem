"""CLI entry point for grounded question answering.

Usage:
    python query.py "what are the key findings?"
    python query.py "..." --top-k 3 --file sample.pdf
    python query.py "..." --doc <doc_id>
    python query.py "..." --retrieval-only      # skip the LLM, dump chunks only
    python query.py "..." --show-chunks         # print chunks alongside the answer

Retrieves the top-k chunks from ChromaDB and (by default) calls Anthropic
to produce a grounded answer with citations.
"""

from __future__ import annotations

import argparse
import logging
import sys

from backend.config import get_settings
from backend.services.generation import generate_answer
from backend.services.retrieval import RetrievedChunk, retrieve


def _print_chunks(chunks: list[RetrievedChunk]) -> None:
    print(f"\nRetrieved {len(chunks)} chunks:")
    for rank, chunk in enumerate(chunks, start=1):
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:240] + "..."
        print(f"  [{rank}] score={chunk.score:.4f}  {chunk.citation}")
        print(f"      {preview}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a grounded question against the RAG corpus.")
    parser.add_argument("question", help="Natural-language question.")
    parser.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve.")
    parser.add_argument("--file", dest="filename", default=None, help="Restrict to a filename.")
    parser.add_argument("--doc", dest="doc_id", default=None, help="Restrict to a doc_id.")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip the LLM call.")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunks too.")
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

    if args.retrieval_only:
        print(f"Query: {args.question}")
        _print_chunks(chunks)
        return 0

    answer = generate_answer(args.question, chunks)

    print(f"Q: {args.question}\n")
    print(f"A: {answer.text}\n")
    print("Citations:")
    for c in answer.citations:
        print(f"  [{c.rank}] {c.filename} (p.{c.page_number}, chunk {c.chunk_index})  score={c.score:.4f}")
    print(
        f"\nModel: {answer.model}  |  tokens in/out: "
        f"{answer.input_tokens}/{answer.output_tokens}"
    )

    if args.show_chunks:
        _print_chunks(chunks)

    return 0


if __name__ == "__main__":
    sys.exit(main())
