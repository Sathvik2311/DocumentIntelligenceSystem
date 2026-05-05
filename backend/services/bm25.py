"""BM25 keyword index over the Chroma collection.

Built lazily from the collection on first query, cached in-process, and invalidated
via `mark_dirty()` whenever ingestion mutates the corpus. Tokenisation is a simple
lowercase + alphanumeric split — good enough for English; can be swapped later.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Module-level cache state. The lock guards rebuilds; reads are best-effort.
_LOCK = threading.Lock()
_INDEX: _BM25Snapshot | None = None
_DIRTY: bool = True

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class _BM25Snapshot:
    """Immutable BM25 index plus the chunk-id list it ranks."""

    index: BM25Okapi
    chunk_ids: list[str]


def mark_dirty() -> None:
    """Flag the in-process BM25 cache for rebuild on the next query."""
    global _DIRTY
    _DIRTY = True


def reset_cache() -> None:
    """Drop the cached index entirely (used by tests for isolation)."""
    global _INDEX, _DIRTY
    with _LOCK:
        _INDEX = None
        _DIRTY = True


def _rebuild_from_collection() -> _BM25Snapshot | None:
    """Pull every chunk from Chroma and build a BM25Okapi over their tokens.

    Returns None if the collection is empty (caller should treat it as "no sparse
    candidates" without crashing).
    """
    # Local import to avoid a circular dep with services.ingestion at module load.
    from backend.services.ingestion import get_collection

    rows = get_collection().get(include=["documents"])
    documents: list[str] = rows.get("documents") or []
    ids: list[str] = rows.get("ids") or []
    if not documents:
        return None

    tokenised = [_tokenize(doc) for doc in documents]
    logger.info("Rebuilt BM25 index over %d chunks", len(documents))
    return _BM25Snapshot(index=BM25Okapi(tokenised), chunk_ids=ids)


def _ensure_fresh() -> _BM25Snapshot | None:
    """Return a current snapshot, rebuilding if dirty."""
    global _INDEX, _DIRTY
    if not _DIRTY and _INDEX is not None:
        return _INDEX
    with _LOCK:
        if not _DIRTY and _INDEX is not None:
            return _INDEX
        _INDEX = _rebuild_from_collection()
        _DIRTY = False
        return _INDEX


def score_query(query: str, top_n: int) -> list[tuple[str, float]]:
    """Return the top-N (chunk_id, bm25_score) pairs ranked highest-first.

    Empty list if the corpus is empty or the query produced no tokens.
    """
    if top_n <= 0:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    snap = _ensure_fresh()
    if snap is None:
        return []
    scores = snap.index.get_scores(tokens)
    # argsort descending on score, drop zero-score hits (BM25 returns 0 for no overlap).
    ranked = sorted(
        ((cid, float(s)) for cid, s in zip(snap.chunk_ids, scores, strict=False) if s > 0),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:top_n]


def filter_ids_by_metadata(
    candidates: list[tuple[str, float]],
    where: dict[str, Any] | None,
) -> list[tuple[str, float]]:
    """Drop candidates whose metadata doesn't match the Chroma `where` clause.

    Implemented as a post-filter rather than rebuilding indices per filter combo —
    fine while the corpus is small (<10k chunks). For larger corpora, prefer
    pushing the filter into Chroma's `get` and rebuilding.
    """
    if not where or not candidates:
        return candidates

    from backend.services.ingestion import get_collection

    ids = [cid for cid, _ in candidates]
    rows = get_collection().get(ids=ids, where=where, include=[])
    kept = set(rows.get("ids") or [])
    return [(cid, s) for cid, s in candidates if cid in kept]
