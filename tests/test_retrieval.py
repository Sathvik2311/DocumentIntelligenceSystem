"""Tests for the retrieval service."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.services.ingestion import ingest_document
from backend.services.retrieval import retrieve


def test_retrieve_returns_top_k(sample_pdf: Path) -> None:
    ingest_document(sample_pdf)
    chunks = retrieve("hello world", top_k=2)
    assert len(chunks) == 2
    # Scores must be sorted desc (similarity, not distance).
    assert chunks[0].score >= chunks[1].score
    # Sanity: cosine similarity in [-1, 1].
    for c in chunks:
        assert -1.0 <= c.score <= 1.0


def test_retrieve_empty_collection() -> None:
    assert retrieve("anything") == []


def test_retrieve_filter_by_filename(sample_pdf: Path, sample_txt: Path) -> None:
    ingest_document(sample_pdf)
    ingest_document(sample_txt)
    only_pdf = retrieve("greeting", top_k=5, filename="sample.pdf")
    assert only_pdf
    assert all(c.filename == "sample.pdf" for c in only_pdf)


def test_retrieve_filter_by_doc_id(sample_pdf: Path, sample_txt: Path) -> None:
    a = ingest_document(sample_pdf)
    b = ingest_document(sample_txt)
    only_a = retrieve("greeting", top_k=5, doc_id=a.doc_id)
    assert only_a
    assert all(c.doc_id == a.doc_id for c in only_a)
    assert all(c.doc_id != b.doc_id for c in only_a)


def test_retrieve_combined_filters(sample_pdf: Path, sample_txt: Path) -> None:
    a = ingest_document(sample_pdf)
    ingest_document(sample_txt)
    chunks = retrieve("greeting", top_k=5, doc_id=a.doc_id, filename="sample.pdf")
    assert all(c.doc_id == a.doc_id and c.filename == "sample.pdf" for c in chunks)


def test_retrieve_no_match_filter(sample_pdf: Path) -> None:
    ingest_document(sample_pdf)
    assert retrieve("anything", filename="nope.pdf") == []


def test_retrieve_empty_query_raises(sample_pdf: Path) -> None:
    ingest_document(sample_pdf)
    with pytest.raises(ValueError, match="non-empty"):
        retrieve("   ")


def test_retrieve_zero_top_k_raises(sample_pdf: Path) -> None:
    ingest_document(sample_pdf)
    with pytest.raises(ValueError, match="positive"):
        retrieve("anything", top_k=0)


def test_retrieved_chunk_citation_string(sample_pdf: Path) -> None:
    ingest_document(sample_pdf)
    [chunk] = retrieve("hello world", top_k=1)
    assert chunk.filename in chunk.citation
    assert "p." in chunk.citation
    assert "chunk" in chunk.citation


# ---------- Multi-doc filter (doc_ids) ----------


def test_retrieve_filter_by_doc_ids_subset(
    sample_pdf: Path, sample_txt: Path, sample_docx: Path
) -> None:
    a = ingest_document(sample_pdf)
    b = ingest_document(sample_txt)
    c = ingest_document(sample_docx)

    chunks = retrieve("dog fox greeting", top_k=10, doc_ids=[a.doc_id, c.doc_id])
    assert chunks
    seen = {ch.doc_id for ch in chunks}
    assert seen <= {a.doc_id, c.doc_id}
    assert b.doc_id not in seen


def test_retrieve_doc_ids_single_element(sample_pdf: Path, sample_txt: Path) -> None:
    a = ingest_document(sample_pdf)
    ingest_document(sample_txt)
    chunks = retrieve("anything", top_k=5, doc_ids=[a.doc_id])
    assert chunks
    assert all(c.doc_id == a.doc_id for c in chunks)


def test_retrieve_doc_ids_empty_means_all(sample_pdf: Path, sample_txt: Path) -> None:
    ingest_document(sample_pdf)
    ingest_document(sample_txt)
    chunks = retrieve("the quick brown fox", top_k=10, doc_ids=[])
    seen = {c.filename for c in chunks}
    assert seen == {"sample.pdf", "sample.txt"}


def test_retrieve_doc_ids_takes_precedence_over_doc_id(
    sample_pdf: Path, sample_txt: Path
) -> None:
    a = ingest_document(sample_pdf)
    b = ingest_document(sample_txt)
    # doc_ids restricts to {b}; doc_id (=a) should be ignored.
    chunks = retrieve("anything", top_k=5, doc_id=a.doc_id, doc_ids=[b.doc_id])
    assert chunks
    assert all(c.doc_id == b.doc_id for c in chunks)


def test_retrieve_doc_ids_no_match_returns_empty(sample_pdf: Path) -> None:
    ingest_document(sample_pdf)
    assert retrieve("anything", doc_ids=["nope-1", "nope-2"]) == []


# ---------- Hybrid search (BM25 + cosine via RRF) ----------


@pytest.fixture
def keyword_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Two TXTs: one mentions a rare proper noun, the other only generic prose."""
    target = tmp_path / "target.txt"
    target.write_text(
        "Quarterly report. Project Zarvox-7 begins operations in Q3. "
        "Deliverables include a status memo and a stakeholder briefing."
    )
    distractor = tmp_path / "distractor.txt"
    distractor.write_text(
        "General overview. The organisation focuses on customer success and "
        "operational excellence across regions and product lines."
    )
    return target, distractor


def test_use_hybrid_true_finds_rare_keyword(
    keyword_corpus: tuple[Path, Path],
) -> None:
    target, distractor = keyword_corpus
    ingest_document(target)
    ingest_document(distractor)
    chunks = retrieve("Zarvox-7", top_k=2, use_hybrid=True)
    assert chunks
    assert chunks[0].filename == "target.txt"


def test_use_hybrid_false_falls_back_to_dense(
    keyword_corpus: tuple[Path, Path],
) -> None:
    """With hybrid off, no exception and only dense results — score is cosine sim."""
    target, distractor = keyword_corpus
    ingest_document(target)
    ingest_document(distractor)
    chunks = retrieve("Zarvox-7", top_k=2, use_hybrid=False)
    assert chunks
    # Dense scores are cosine similarity; BM25/RRF-fused scores are tiny RRF values.
    # Dense tops out close to ~0.5+ for at-least-vaguely-related text; assert > 0.05.
    assert chunks[0].score > 0.05


def test_use_hybrid_skips_bm25_when_disabled(
    keyword_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies BM25.score_query is not called when use_hybrid=False."""
    from backend.services import bm25, retrieval

    target, distractor = keyword_corpus
    ingest_document(target)
    ingest_document(distractor)

    calls: list[tuple[str, int]] = []

    def spy(query: str, top_n: int) -> list[tuple[str, float]]:
        calls.append((query, top_n))
        return []

    monkeypatch.setattr(bm25, "score_query", spy)
    # Also patch the alias retrieval.py imported.
    monkeypatch.setattr(retrieval.bm25, "score_query", spy)

    retrieve("anything", top_k=2, use_hybrid=False)
    assert calls == []


def test_hybrid_respects_filename_filter(
    keyword_corpus: tuple[Path, Path],
) -> None:
    target, distractor = keyword_corpus
    ingest_document(target)
    ingest_document(distractor)
    chunks = retrieve(
        "Zarvox-7",
        top_k=5,
        use_hybrid=True,
        filename="distractor.txt",
    )
    # The rare keyword lives only in target.txt; filtering to distractor returns nothing
    # (or unrelated chunks with score 0). Either way, no target.txt rows.
    assert all(c.filename != "target.txt" for c in chunks)
