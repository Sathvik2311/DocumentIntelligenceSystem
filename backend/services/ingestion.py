"""Ingestion pipeline: parse documents and chunk them into token-bounded pieces.

Day 2 scope: parsing (PDF/DOCX/TXT) + token-based chunking with overlap.
Embedding + ChromaDB persistence land in Day 3.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import tiktoken
from docx import Document as DocxDocument

from backend.config import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class ParsedPage:
    """A single page of parsed text."""

    text: str
    page_number: int


@dataclass(frozen=True)
class DocumentChunk:
    """A token-bounded chunk ready for embedding."""

    text: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    token_count: int

    def metadata(self) -> dict[str, str | int]:
        """ChromaDB-compatible metadata dict (scalar values only)."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed document with identity and page list."""

    doc_id: str
    filename: str
    pages: list[ParsedPage] = field(default_factory=list)


def parse_document(path: str | Path, doc_id: str | None = None) -> ParsedDocument:
    """Parse a PDF, DOCX, or TXT file into a ParsedDocument.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: extension is not supported.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension {ext!r}; expected one of {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        pages = _parse_pdf(p)
    elif ext == ".docx":
        pages = _parse_docx(p)
    else:
        pages = _parse_txt(p)

    return ParsedDocument(
        doc_id=doc_id or uuid.uuid4().hex,
        filename=p.name,
        pages=pages,
    )


def _parse_pdf(path: Path) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(ParsedPage(text=text, page_number=i))
    return pages


def _parse_docx(path: Path) -> list[ParsedPage]:
    # python-docx does not expose true page breaks from the XML, so the whole
    # document is surfaced as page 1. Upstream consumers should not depend on
    # page-level granularity for DOCX inputs.
    doc = DocxDocument(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [ParsedPage(text=text, page_number=1)] if text else []


def _parse_txt(path: Path) -> list[ParsedPage]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [ParsedPage(text=text, page_number=1)] if text else []


def chunk_document(
    document: ParsedDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Split a parsed document into token-bounded chunks with overlap.

    Chunks never cross page boundaries, so each chunk carries the page number
    it came from. Sizes default to the values in `backend.config.Settings`.
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    step = size - overlap
    chunks: list[DocumentChunk] = []
    global_index = 0

    for page in document.pages:
        tokens = _ENCODING.encode(page.text)
        if not tokens:
            continue
        for start in range(0, len(tokens), step):
            window = tokens[start : start + size]
            if not window:
                break
            chunks.append(
                DocumentChunk(
                    text=_ENCODING.decode(window),
                    doc_id=document.doc_id,
                    filename=document.filename,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    token_count=len(window),
                )
            )
            global_index += 1
            if start + size >= len(tokens):
                break

    logger.info(
        "Chunked %s: %d pages -> %d chunks (size=%d overlap=%d)",
        document.filename,
        len(document.pages),
        len(chunks),
        size,
        overlap,
    )
    return chunks


def ingest_file(path: str | Path) -> list[DocumentChunk]:
    """Parse + chunk a file in one call. Convenience wrapper for Day 2."""
    return chunk_document(parse_document(path))
