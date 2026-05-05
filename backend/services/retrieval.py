"""Retrieval service: embed a query and pull the top-k similar chunks from ChromaDB.

Pipeline (each stage is independently toggleable):

    question
       │
       ├── dense:  embed(query) → Chroma cosine top-N
       │
       └── sparse: BM25 over chunk text → top-N        (when use_hybrid)
                            │
                            ▼
                Reciprocal Rank Fusion (RRF, k=60)
                            │
                            ▼
                fused candidates → top-K return
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.config import get_settings
from backend.services import bm25
from backend.services.ingestion import embed_texts, get_collection

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF dampening constant


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from similarity search.

    `score` semantics depend on the active pipeline:
      • Dense-only retrieval:  cosine similarity in [-1, 1].
      • Hybrid (no rerank):    Reciprocal Rank Fusion score (unbounded, small positive).
      • Reranker active:       raw cross-encoder logit (unbounded; higher = more relevant).
    """

    text: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float

    @property
    def citation(self) -> str:
        """Short human-readable citation string: `filename (p.N, chunk K)`."""
        return f"{self.filename} (p.{self.page_number}, chunk {self.chunk_index})"


def retrieve(
    query: str,
    top_k: int | None = None,
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
    filename: str | None = None,
    use_hybrid: bool | None = None,
    use_reranker: bool | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k chunks most similar to `query`.

    Args:
        query: Natural-language question.
        top_k: Number of chunks to return. Defaults to settings.top_k.
        doc_id: Optional filter — restrict to a single document by id (legacy).
        doc_ids: Optional filter — restrict to a specific set of documents.
            Takes precedence over `doc_id` when both are set.
        filename: Optional filter — restrict to a single document by filename.
        use_hybrid: Override ENABLE_HYBRID_SEARCH per request. None = default.
        use_reranker: Override ENABLE_RERANKER per request. None = default.

    Returns:
        A list of RetrievedChunk sorted by descending score. Empty list if the
        collection is empty or nothing matches the filter.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    if k <= 0:
        raise ValueError("top_k must be positive")

    # Normalise the doc filter: prefer doc_ids; fall back to doc_id.
    effective_ids: list[str] | None = None
    if doc_ids:
        effective_ids = [d for d in doc_ids if d]
    elif doc_id:
        effective_ids = [doc_id]

    collection = get_collection()
    total = collection.count()
    if total == 0:
        logger.warning("retrieval called on empty collection")
        return []

    where = _build_where(doc_ids=effective_ids, filename=filename)

    hybrid_on = settings.enable_hybrid_search if use_hybrid is None else use_hybrid
    rerank_on = settings.enable_reranker if use_reranker is None else use_reranker
    candidate_n = max(k, settings.retrieval_candidates)

    # --- Stage 1: dense (always on) ---
    dense_chunks = _dense_search(query, candidate_n, where, total)

    if hybrid_on:
        # --- Stage 2: sparse (BM25) ---
        sparse_pairs = bm25.score_query(query, top_n=candidate_n)
        sparse_pairs = bm25.filter_ids_by_metadata(sparse_pairs, where)
        # --- Stage 3: Reciprocal Rank Fusion ---
        candidates = _fuse_with_rrf(dense_chunks, sparse_pairs, where=where, top_n=candidate_n)
    else:
        candidates = dense_chunks[:candidate_n]

    if rerank_on and len(candidates) > 1:
        # --- Stage 4: Cross-encoder rerank ---
        # Lazy import to avoid loading the model when rerank is disabled.
        from backend.services.reranker import rerank as _rerank
        return _rerank(query, candidates, top_k=k)

    return candidates[:k]


# ---------- Pipeline stages ----------


def _dense_search(
    query: str,
    n_results: int,
    where: dict[str, Any] | None,
    total: int,
) -> list[RetrievedChunk]:
    """Cosine similarity search via Chroma. Returns chunks ranked best-first."""
    query_embedding = embed_texts([query])
    result = get_collection().query(
        query_embeddings=query_embedding,
        n_results=min(n_results, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = (result.get("ids") or [[]])[0]
    return _rows_to_chunks(result, ids=ids)


def _fuse_with_rrf(
    dense: list[RetrievedChunk],
    sparse: list[tuple[str, float]],
    where: dict[str, Any] | None,
    top_n: int,
) -> list[RetrievedChunk]:
    """Combine dense + sparse rankings with Reciprocal Rank Fusion.

    Score for a candidate = Σ_{stage} 1 / (k + rank_in_stage). Candidates missing
    from a stage simply don't contribute that stage's term.
    """
    rrf_scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense):
        cid = _chunk_id(chunk)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunks_by_id[cid] = chunk

    # Sparse candidates may not be in the dense set; hydrate them via Chroma.
    sparse_ids_to_hydrate = [cid for cid, _ in sparse if cid not in chunks_by_id]
    if sparse_ids_to_hydrate:
        for chunk in _hydrate_chunks_by_id(sparse_ids_to_hydrate):
            chunks_by_id[_chunk_id(chunk)] = chunk

    for rank, (cid, _bm25_score) in enumerate(sparse):
        if cid not in chunks_by_id:
            # Filtered out by the where clause during hydration; skip.
            continue
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

    fused = sorted(
        (
            _replace_score(chunks_by_id[cid], rrf_scores[cid])
            for cid in rrf_scores
            if cid in chunks_by_id
        ),
        key=lambda c: c.score,
        reverse=True,
    )
    return fused[:top_n]


def _hydrate_chunks_by_id(ids: list[str]) -> list[RetrievedChunk]:
    """Fetch chunks by id from Chroma so we can build RetrievedChunk objects."""
    if not ids:
        return []
    rows = get_collection().get(ids=ids, include=["documents", "metadatas"])
    out: list[RetrievedChunk] = []
    fetched_ids = rows.get("ids") or []
    documents = rows.get("documents") or []
    metadatas = rows.get("metadatas") or []
    for _cid, text, meta in zip(fetched_ids, documents, metadatas, strict=False):
        out.append(
            RetrievedChunk(
                text=text,
                doc_id=str(meta.get("doc_id", "")),
                filename=str(meta.get("filename", "")),
                page_number=int(meta.get("page_number", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=0.0,  # placeholder; overwritten in fusion
            )
        )
    return out


def _chunk_id(chunk: RetrievedChunk) -> str:
    """Canonical id matching `store_chunks()`'s ingestion key format."""
    return f"{chunk.doc_id}:{chunk.chunk_index}"


def _replace_score(chunk: RetrievedChunk, new_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=chunk.text,
        doc_id=chunk.doc_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        score=new_score,
    )


# ---------- Helpers ----------


def _build_where(
    doc_ids: list[str] | None,
    filename: str | None,
) -> dict[str, Any] | None:
    """Translate optional filters into a Chroma `where` clause."""
    clauses: list[dict[str, Any]] = []
    if doc_ids:
        if len(doc_ids) == 1:
            clauses.append({"doc_id": doc_ids[0]})
        else:
            clauses.append({"doc_id": {"$in": list(doc_ids)}})
    if filename:
        clauses.append({"filename": filename})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _rows_to_chunks(
    result: dict[str, Any], ids: list[str] | None = None
) -> list[RetrievedChunk]:
    """Flatten a Chroma `query` result (first query only) into RetrievedChunk objects."""
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for text, meta, distance in zip(documents, metadatas, distances, strict=False):
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
