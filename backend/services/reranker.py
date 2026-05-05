"""Cross-encoder reranker.

Reads a (query, candidate-text) pair *together* and returns a relevance score —
much more accurate than the bi-encoder cosine score the dense retriever uses,
at the cost of being slower per pair. Run on a small candidate set (top-N from
hybrid search) and keep the top-K best.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Lazily load and cache the configured cross-encoder."""
    name = get_settings().reranker_model
    logger.info("Loading reranker model: %s", name)
    return CrossEncoder(name)


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Rerank `candidates` by cross-encoder relevance and return the top_k.

    The returned chunks have their `score` field replaced with the cross-encoder
    logit (higher = more relevant). Unbounded; values typically in [-10, 10].
    """
    if not candidates or top_k <= 0:
        return []
    if len(candidates) == 1:
        return list(candidates)

    pairs = [(query, c.text) for c in candidates]
    scores = get_reranker().predict(pairs)

    scored: list[RetrievedChunk] = [
        RetrievedChunk(
            text=c.text,
            doc_id=c.doc_id,
            filename=c.filename,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            score=float(s),
        )
        for c, s in zip(candidates, scores, strict=False)
    ]
    scored.sort(key=lambda ch: ch.score, reverse=True)
    return scored[:top_k]
