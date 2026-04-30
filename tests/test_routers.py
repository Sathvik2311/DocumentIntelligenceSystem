"""End-to-end FastAPI tests via TestClient. The LLM provider is monkeypatched so
no real network calls happen."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import generation


class _FakeProvider:
    def complete(self, system: str, messages: list[dict]) -> generation.ProviderResponse:
        self.last_messages = messages
        return generation.ProviderResponse(
            text="ok [1]",
            model="fake",
            input_tokens=10,
            output_tokens=2,
        )


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> _FakeProvider:
    fake = _FakeProvider()
    generation._get_provider.cache_clear()
    monkeypatch.setattr(generation, "_get_provider", lambda: fake)
    return fake


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------- Health ----------


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------- Documents CRUD ----------


def test_list_empty(client: TestClient) -> None:
    r = client.get("/api/documents")
    assert r.status_code == 200
    assert r.json() == {"documents": []}


def test_upload_list_delete_cycle(client: TestClient, sample_pdf: Path) -> None:
    with sample_pdf.open("rb") as f:
        r = client.post(
            "/api/documents/upload",
            files={"file": ("sample.pdf", f, "application/pdf")},
        )
    assert r.status_code == 201, r.text
    body = r.json()
    doc_id = body["doc_id"]
    assert body["filename"] == "sample.pdf"
    assert body["num_chunks"] > 0

    r = client.get("/api/documents")
    assert r.status_code == 200
    listed = r.json()["documents"]
    assert any(d["doc_id"] == doc_id for d in listed)

    r = client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["deleted_chunks"] == body["num_chunks"]

    r = client.get("/api/documents")
    assert all(d["doc_id"] != doc_id for d in r.json()["documents"])


def test_upload_unsupported_extension_returns_400(
    client: TestClient, tmp_path: Path
) -> None:
    md = tmp_path / "x.md"
    md.write_text("hi")
    with md.open("rb") as f:
        r = client.post(
            "/api/documents/upload",
            files={"file": ("x.md", f, "text/markdown")},
        )
    assert r.status_code == 400
    body = r.json()
    assert body["status_code"] == 400
    assert "Unsupported" in body["message"]


def test_delete_unknown_returns_404(client: TestClient) -> None:
    r = client.delete("/api/documents/does-not-exist")
    assert r.status_code == 404
    assert r.json()["status_code"] == 404


# ---------- Query ----------


def _upload(client: TestClient, path: Path) -> str:
    with path.open("rb") as f:
        r = client.post(
            "/api/documents/upload",
            files={"file": (path.name, f, "application/octet-stream")},
        )
    assert r.status_code == 201, r.text
    return r.json()["doc_id"]


def test_query_validation_error_on_empty_question(client: TestClient) -> None:
    r = client.post("/api/query", json={"question": ""})
    assert r.status_code == 422
    assert r.json()["status_code"] == 422


def test_query_retrieval_only_skips_llm(
    client: TestClient, sample_pdf: Path
) -> None:
    _upload(client, sample_pdf)
    r = client.post(
        "/api/query",
        json={"question": "hello world", "retrieval_only": True, "top_k": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "(none)"
    assert body["input_tokens"] == 0 and body["output_tokens"] == 0
    assert body["citations"], "retrieval-only should still return chunks"
    assert all(c.get("text") for c in body["citations"])


def test_query_no_match_filter_returns_no_context_answer(
    client: TestClient, sample_pdf: Path
) -> None:
    _upload(client, sample_pdf)
    r = client.post(
        "/api/query",
        json={"question": "anything", "retrieval_only": True, "filename": "nope.pdf"},
    )
    assert r.status_code == 200
    assert r.json()["citations"] == []


def test_query_full_path_calls_provider(
    client: TestClient, fake_provider: _FakeProvider, sample_pdf: Path
) -> None:
    _upload(client, sample_pdf)
    r = client.post(
        "/api/query",
        json={"question": "what is on each page?", "top_k": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "ok [1]"
    assert body["model"] == "fake"
    assert body["input_tokens"] == 10
    assert body["output_tokens"] == 2
    assert len(body["citations"]) == 2
    assert all(c.get("text") for c in body["citations"])


def test_query_history_is_forwarded_to_provider(
    client: TestClient, fake_provider: _FakeProvider, sample_pdf: Path
) -> None:
    _upload(client, sample_pdf)
    r = client.post(
        "/api/query",
        json={
            "question": "follow up",
            "top_k": 1,
            "history": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
            ],
        },
    )
    assert r.status_code == 200, r.text
    msgs = fake_provider.last_messages
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "first"
    assert msgs[1]["content"] == "reply"


def test_query_invalid_history_role_returns_422(
    client: TestClient, sample_pdf: Path
) -> None:
    _upload(client, sample_pdf)
    r = client.post(
        "/api/query",
        json={
            "question": "q",
            "history": [{"role": "system", "content": "bad"}],
        },
    )
    assert r.status_code == 422


def test_query_doc_ids_restricts_to_subset(
    client: TestClient, sample_pdf: Path, sample_txt: Path
) -> None:
    pdf_id = _upload(client, sample_pdf)
    _upload(client, sample_txt)

    r = client.post(
        "/api/query",
        json={
            "question": "anything",
            "retrieval_only": True,
            "top_k": 5,
            "doc_ids": [pdf_id],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["citations"]
    assert all(c["filename"] == "sample.pdf" for c in body["citations"])


def test_query_empty_doc_ids_searches_all(
    client: TestClient, sample_pdf: Path, sample_txt: Path
) -> None:
    _upload(client, sample_pdf)
    _upload(client, sample_txt)

    r = client.post(
        "/api/query",
        json={
            "question": "the quick brown fox",
            "retrieval_only": True,
            "top_k": 10,
            "doc_ids": [],
        },
    )
    assert r.status_code == 200, r.text
    filenames = {c["filename"] for c in r.json()["citations"]}
    assert filenames == {"sample.pdf", "sample.txt"}
