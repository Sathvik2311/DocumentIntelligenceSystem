"""Retrieval service: embed a query and pull the top-k similar chunks from ChromaDB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.config import get_settings
from backend.services.ingestion import embed_texts, get_collection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from similarity search, with its cosine similarity score."""

    text: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float  # cosine similarity in [-1, 1]; typically [0, 1] for this embedder.

    @property
    def citation(self) -> str:
        """Short human-readable citation string: `filename (p.N, chunk K)`."""
        return f"{self.filename} (p.{self.page_number}, chunk {self.chunk_index})"


def retrieve(
    query: str,
    top_k: int | None = None,
    doc_id: str | None = None,
    filename: str | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k chunks most similar to `query`.

    Args:
        query: Natural-language question.
        top_k: Number of chunks to return. Defaults to settings.top_k.
        doc_id: Optional filter — restrict search to a single document by id.
        filename: Optional filter — restrict search to a single document by filename.

    Returns:
        A list of RetrievedChunk sorted by descending similarity. Empty list if the
        collection is empty or nothing matches the filter.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    k = top_k if top_k is not None else get_settings().top_k
    if k <= 0:
        raise ValueError("top_k must be positive")

    collection = get_collection()
    total = collection.count()
    if total == 0:
        logger.warning("retrieval called on empty collection")
        return []

    where = _build_where(doc_id=doc_id, filename=filename)
    query_embedding = embed_texts([query])

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return _rows_to_chunks(result)


def _build_where(doc_id: str | None, filename: str | None) -> dict[str, Any] | None:
    """Translate optional filters into a Chroma `where` clause."""
    clauses: list[dict[str, Any]] = []
    if doc_id:
        clauses.append({"doc_id": doc_id})
    if filename:
        clauses.append({"filename": filename})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _rows_to_chunks(result: dict[str, Any]) -> list[RetrievedChunk]:
    """Flatten a Chroma query result (first query only) into RetrievedChunk objects."""
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                text=text,
                doc_id=str(meta.get("doc_id", "")),
                filename=str(meta.get("filename", "")),
                page_number=int(meta.get("page_number", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=1.0 - float(distance),  # cosine distance -> similarity
            )
        )
    return chunks
