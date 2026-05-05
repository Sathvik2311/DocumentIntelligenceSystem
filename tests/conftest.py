"""Shared pytest fixtures: isolated Chroma store, sample files, mocked LLM provider."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from docx import Document as DocxDocument

# Make the project root importable when pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point Chroma at a fresh per-test directory and reset all module-level caches."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")  # default; tests that hit LLM mock the provider
    # Don't trigger the real cross-encoder download in CI; reranker tests opt in.
    monkeypatch.setenv("ENABLE_RERANKER", "false")

    # Drop cached settings + module-level Chroma client/collection/embedder so each test
    # builds them from scratch against the new directory.
    from backend import config as _config
    from backend.services import (
        bm25 as _bm25,
    )
    from backend.services import (
        generation as _gen,
    )
    from backend.services import (
        ingestion as _ing,
    )
    from backend.services import (
        reranker as _rr,
    )

    _config.get_settings.cache_clear()
    _gen._get_provider.cache_clear()
    _rr.get_reranker.cache_clear()
    _ing._CHROMA_CLIENT = None
    _ing._COLLECTION = None
    _bm25.reset_cache()
    # Keep the embedder cached across tests — it's expensive to load and is read-only.
    yield


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text("The quick brown fox jumps over the lazy dog. " * 80)
    return p


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pdf"
    pdf = pymupdf.open()
    for i in range(3):
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            f"Page {i+1}: hello world. " * 40,
            fontsize=11,
        )
    pdf.save(p)
    pdf.close()
    return p


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    p = tmp_path / "sample.docx"
    doc = DocxDocument()
    for i in range(20):
        doc.add_paragraph(f"Paragraph {i}: the quick brown fox jumps over the lazy dog. " * 5)
    doc.save(p)
    return p
