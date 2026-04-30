"""Tests for the cross-encoder reranker. Uses a fake CrossEncoder so the real
~22 MB model never has to load in CI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.services import reranker as reranker_mod
from backend.services.ingestion import ingest_document
from backend.services.reranker import rerank
from backend.services.retrieval import RetrievedChunk, retrieve


class _FakeCrossEncoder:
    """Returns scores in reverse order of the input pairs (last → highest score)."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        # First pair gets lowest score, last gets highest.
        return [float(i) for i in range(len(pairs))]


@pytest.fixture
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> _FakeCrossEncoder:
    fake = _FakeCrossEncoder()
    reranker_mod.get_reranker.cache_clear()
    monkeypatch.setattr(reranker_mod, "get_reranker", lambda: fake)
    return fake


def _chunk(score: float, idx: int, text: str = "x") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        doc_id="d",
        filename="f.txt",
        page_number=1,
        chunk_index=idx,
        score=score,
    )


# ---------- Unit tests ----------


def test_rerank_reorders_by_cross_encoder_score(
    fake_reranker: _FakeCrossEncoder,
) -> None:
    # Original order: a (0.9), b (0.5), c (0.1).
    # Fake CE gives last-input the highest score, so reranked order is c, b, a.
    a = _chunk(0.9, 0, "alpha")
    b = _chunk(0.5, 1, "beta")
    c = _chunk(0.1, 2, "gamma")
    out = rerank("q", [a, b, c], top_k=3)
    assert [ch.chunk_index for ch in out] == [2, 1, 0]
    # Scores are now the cross-encoder logits, not the originals.
    assert out[0].score > out[1].score > out[2].score


def test_rerank_top_k_truncates(fake_reranker: _FakeCrossEncoder) -> None:
    chunks = [_chunk(0.0, i) for i in range(5)]
    out = rerank("q", chunks, top_k=2)
    assert len(out) == 2


def test_rerank_empty_input(fake_reranker: _FakeCrossEncoder) -> None:
    assert rerank("q", [], top_k=5) == []


def test_rerank_single_candidate_skips_model(
    fake_reranker: _FakeCrossEncoder,
) -> None:
    """No need to invoke the model for a single candidate."""
    only = _chunk(0.42, 0)
    out = rerank("q", [only], top_k=5)
    assert out == [only]
    assert fake_reranker.calls == []


# ---------- Integration with retrieve() ----------


def test_use_reranker_true_calls_cross_encoder(
    fake_reranker: _FakeCrossEncoder, sample_pdf: Path
) -> None:
    ingest_document(sample_pdf)
    chunks = retrieve("hello world", top_k=2, use_reranker=True)
    assert chunks
    assert fake_reranker.calls, "expected the cross-encoder to be invoked"


def test_use_reranker_false_skips_cross_encoder(
    fake_reranker: _FakeCrossEncoder, sample_pdf: Path
) -> None:
    ingest_document(sample_pdf)
    retrieve("hello world", top_k=2, use_reranker=False)
    assert fake_reranker.calls == []


def test_reranker_changes_ordering_in_full_pipeline(
    fake_reranker: _FakeCrossEncoder, sample_pdf: Path
) -> None:
    """The fake CE flips ordering; verify the final retrieve() output reflects that."""
    ingest_document(sample_pdf)
    without = retrieve("hello world", top_k=3, use_reranker=False)
    with_rr = retrieve("hello world", top_k=3, use_reranker=True)

    # If at least 2 candidates exist, the rerank order should differ from the
    # un-reranked order (since the fake CE reverses by input position).
    if len(without) >= 2:
        assert [c.chunk_index for c in with_rr] != [c.chunk_index for c in without]