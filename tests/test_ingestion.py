"""Tests for the ingestion pipeline: parsing, chunking, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.services.ingestion import (
    SUPPORTED_EXTENSIONS,
    chunk_document,
    delete_document,
    get_collection,
    ingest_document,
    list_documents,
    parse_document,
)

# ---------- parse_document ----------


def test_parse_txt(sample_txt: Path) -> None:
    doc = parse_document(sample_txt)
    assert doc.filename == "sample.txt"
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert "fox" in doc.pages[0].text


def test_parse_docx(sample_docx: Path) -> None:
    doc = parse_document(sample_docx)
    assert doc.filename == "sample.docx"
    # python-docx surfaces everything as a single page.
    assert len(doc.pages) == 1
    assert "Paragraph 0" in doc.pages[0].text


def test_parse_pdf_preserves_pages(sample_pdf: Path) -> None:
    doc = parse_document(sample_pdf)
    assert doc.filename == "sample.pdf"
    assert len(doc.pages) == 3
    pages = sorted(p.page_number for p in doc.pages)
    assert pages == [1, 2, 3]
    assert all("hello world" in p.text for p in doc.pages)


def test_parse_unsupported_extension_raises(tmp_path: Path) -> None:
    md = tmp_path / "x.md"
    md.write_text("# hi")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_document(md)


def test_parse_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_document(tmp_path / "does_not_exist.txt")


def test_supported_extensions_constant() -> None:
    assert SUPPORTED_EXTENSIONS == {".pdf", ".docx", ".txt"}


# ---------- chunk_document ----------


def test_chunk_respects_size_and_overlap(sample_txt: Path) -> None:
    doc = parse_document(sample_txt)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    # All chunks except the last should be at the size cap.
    assert all(c.token_count <= 200 for c in chunks)
    assert all(c.token_count > 0 for c in chunks)
    # chunk_index increments globally.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_carry_metadata(sample_pdf: Path) -> None:
    doc = parse_document(sample_pdf)
    chunks = chunk_document(doc)
    assert chunks
    for c in chunks:
        assert c.doc_id == doc.doc_id
        assert c.filename == "sample.pdf"
        assert c.page_number in (1, 2, 3)
        meta = c.metadata()
        assert set(meta.keys()) == {
            "doc_id", "filename", "page_number", "chunk_index", "token_count"
        }


def test_chunks_never_cross_page_boundary(sample_pdf: Path) -> None:
    doc = parse_document(sample_pdf)
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)
    # Each chunk's page should match a single source page; never a mix.
    for c in chunks:
        assert c.page_number in {p.page_number for p in doc.pages}


def test_chunk_invalid_overlap_raises(sample_txt: Path) -> None:
    doc = parse_document(sample_txt)
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(doc, chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(doc, chunk_size=100, chunk_overlap=200)


def test_chunk_invalid_size_raises(sample_txt: Path) -> None:
    doc = parse_document(sample_txt)
    with pytest.raises(ValueError, match="positive"):
        chunk_document(doc, chunk_size=0, chunk_overlap=0)


# ---------- ingest_document + Chroma persistence ----------


def test_ingest_document_persists(sample_txt: Path) -> None:
    result = ingest_document(sample_txt)
    assert result.filename == "sample.txt"
    assert result.num_chunks > 0
    assert get_collection().count() == result.num_chunks


def test_list_documents_groups_by_doc(sample_txt: Path, sample_pdf: Path) -> None:
    a = ingest_document(sample_txt)
    b = ingest_document(sample_pdf)

    docs = list_documents()
    by_id = {d.doc_id: d for d in docs}
    assert a.doc_id in by_id
    assert b.doc_id in by_id
    assert by_id[a.doc_id].num_chunks == a.num_chunks
    assert by_id[b.doc_id].num_chunks == b.num_chunks
    assert by_id[b.doc_id].num_pages == 3  # PDF


def test_list_documents_empty() -> None:
    assert list_documents() == []


def test_delete_document_removes_only_that_doc(
    sample_txt: Path, sample_pdf: Path
) -> None:
    a = ingest_document(sample_txt)
    b = ingest_document(sample_pdf)
    total = a.num_chunks + b.num_chunks
    assert get_collection().count() == total

    deleted = delete_document(a.doc_id)
    assert deleted == a.num_chunks
    assert get_collection().count() == b.num_chunks
    remaining_ids = {d.doc_id for d in list_documents()}
    assert remaining_ids == {b.doc_id}


def test_delete_unknown_doc_returns_zero(sample_txt: Path) -> None:
    ingest_document(sample_txt)
    assert delete_document("not-a-real-id") == 0


# ---------- Auto-summary ----------


class _SummaryFakeProvider:
    """Captures the system + messages and returns a canned summary."""

    def __init__(self, text: str = "This is a TL;DR.") -> None:
        self.text = text
        self.calls: list[tuple[str, list[dict]]] = []

    def complete(self, system, messages):
        from backend.services.generation import ProviderResponse
        self.calls.append((system, list(messages)))
        return ProviderResponse(text=self.text, model="fake", input_tokens=5, output_tokens=3)

    def stream(self, system, messages):  # pragma: no cover — unused for summary
        from backend.services.generation import StreamEvent
        yield StreamEvent(done=True, model="fake")


@pytest.fixture
def summary_provider(monkeypatch: pytest.MonkeyPatch) -> _SummaryFakeProvider:
    from backend.services import generation
    fake = _SummaryFakeProvider()
    generation._get_provider.cache_clear()
    monkeypatch.setattr(generation, "_get_provider", lambda: fake)
    monkeypatch.setenv("ENABLE_AUTO_SUMMARY", "true")
    # Settings is cached; clear so the new env var takes effect.
    from backend import config as _config
    _config.get_settings.cache_clear()
    return fake


def test_ingest_persists_summary_when_enabled(
    summary_provider: _SummaryFakeProvider, sample_pdf: Path
) -> None:
    from backend.services.ingestion import get_summary

    result = ingest_document(sample_pdf)
    assert result.summary == "This is a TL;DR."
    assert get_summary(result.doc_id) == "This is a TL;DR."
    # Summary system prompt was used (not the QA prompt).
    assert summary_provider.calls
    system, _msgs = summary_provider.calls[0]
    assert "summarizer" in system.lower()


def test_list_documents_includes_summary(
    summary_provider: _SummaryFakeProvider, sample_pdf: Path
) -> None:
    result = ingest_document(sample_pdf)
    docs = list_documents()
    found = next(d for d in docs if d.doc_id == result.doc_id)
    assert found.summary == "This is a TL;DR."


def test_delete_removes_summary(
    summary_provider: _SummaryFakeProvider, sample_pdf: Path
) -> None:
    from backend.services.ingestion import get_summary

    result = ingest_document(sample_pdf)
    assert get_summary(result.doc_id) is not None
    delete_document(result.doc_id)
    assert get_summary(result.doc_id) is None


def test_summary_disabled_skips_provider(sample_pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Auto-summary is OFF by default in conftest. Confirm provider is never invoked.
    from backend.services import generation
    from backend.services.ingestion import get_summary

    fake = _SummaryFakeProvider()
    generation._get_provider.cache_clear()
    monkeypatch.setattr(generation, "_get_provider", lambda: fake)

    result = ingest_document(sample_pdf)
    assert result.summary is None
    assert get_summary(result.doc_id) is None
    assert fake.calls == []


def test_summary_provider_failure_does_not_block_ingestion(
    sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.services import generation
    from backend.services.ingestion import get_summary

    class _Boom:
        def complete(self, system, messages):
            raise RuntimeError("provider down")

        def stream(self, system, messages):  # pragma: no cover
            raise RuntimeError("provider down")

    generation._get_provider.cache_clear()
    monkeypatch.setattr(generation, "_get_provider", lambda: _Boom())
    monkeypatch.setenv("ENABLE_AUTO_SUMMARY", "true")
    from backend import config as _config
    _config.get_settings.cache_clear()

    result = ingest_document(sample_pdf)
    assert result.num_chunks > 0  # ingestion succeeded
    assert result.summary is None
    assert get_summary(result.doc_id) is None
