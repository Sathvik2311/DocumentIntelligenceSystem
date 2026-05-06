"""Tests for the generation service. The LLM provider is monkeypatched so we never
hit a real network call."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.services import generation
from backend.services.generation import (
    NO_CONTEXT_ANSWER,
    Answer,
    Message,
    ProviderResponse,
    StreamEvent,
    _build_messages,
    _build_user_message,
    _format_context,
    generate_answer,
    generate_answer_stream,
)
from backend.services.ingestion import ingest_document
from backend.services.retrieval import RetrievedChunk, retrieve


class _FakeProvider:
    """Captures the system + messages it was called with and returns a canned reply."""

    def __init__(self) -> None:
        self.last_system: str | None = None
        self.last_messages: list[Message] | None = None

    def complete(self, system: str, messages: list[Message]) -> ProviderResponse:
        self.last_system = system
        self.last_messages = messages
        return ProviderResponse(
            text="canned answer [1]",
            model="fake-model",
            input_tokens=42,
            output_tokens=7,
        )

    def stream(self, system: str, messages: list[Message]):
        self.last_system = system
        self.last_messages = messages
        for token in ["canned ", "answer ", "[1]"]:
            yield StreamEvent(delta=token)
        yield StreamEvent(
            done=True, model="fake-model", input_tokens=42, output_tokens=7
        )


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> _FakeProvider:
    fake = _FakeProvider()
    # _get_provider is lru_cached; replace the cached value directly.
    generation._get_provider.cache_clear()
    monkeypatch.setattr(generation, "_get_provider", lambda: fake)
    return fake


# ---------- Empty-context short-circuit ----------


def test_generate_empty_chunks_skips_provider(fake_provider: _FakeProvider) -> None:
    answer = generate_answer("anything", chunks=[])
    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.citations == []
    assert answer.input_tokens == 0
    assert answer.output_tokens == 0
    # Provider was never invoked.
    assert fake_provider.last_messages is None


def test_generate_empty_question_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_answer("   ", chunks=[])


# ---------- Happy path ----------


def test_generate_with_chunks_calls_provider(
    fake_provider: _FakeProvider, sample_pdf: Path
) -> None:
    ingest_document(sample_pdf)
    chunks = retrieve("hello world", top_k=2)
    assert chunks

    answer = generate_answer("what is on each page?", chunks)

    assert isinstance(answer, Answer)
    assert answer.text == "canned answer [1]"
    assert answer.model == "fake-model"
    assert answer.input_tokens == 42
    assert answer.output_tokens == 7
    assert len(answer.citations) == len(chunks)
    # System prompt + a single user turn (no history).
    assert fake_provider.last_messages is not None
    assert [m["role"] for m in fake_provider.last_messages] == ["user"]
    assert "Context:" in fake_provider.last_messages[0]["content"]
    assert "Question: what is on each page?" in fake_provider.last_messages[0]["content"]


# ---------- Conversation history ----------


def test_history_is_replayed_before_new_question(
    fake_provider: _FakeProvider, sample_pdf: Path
) -> None:
    ingest_document(sample_pdf)
    chunks = retrieve("hello world", top_k=1)
    history: list[Message] = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    generate_answer("follow up", chunks, history=history)

    msgs = fake_provider.last_messages
    assert msgs is not None
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "earlier question"
    assert msgs[1]["content"] == "earlier answer"
    assert "follow up" in msgs[2]["content"]


def test_history_skips_invalid_turns(
    fake_provider: _FakeProvider, sample_pdf: Path
) -> None:
    ingest_document(sample_pdf)
    chunks = retrieve("hello world", top_k=1)
    history = [
        {"role": "system", "content": "ignored"},   # bad role
        {"role": "user", "content": ""},            # empty content
        {"role": "user", "content": "kept"},
    ]
    generate_answer("q", chunks, history=history)  # type: ignore[arg-type]

    msgs = fake_provider.last_messages
    assert msgs is not None
    # Only "kept" + the new user turn should survive.
    assert [m["role"] for m in msgs] == ["user", "user"]
    assert msgs[0]["content"] == "kept"


# ---------- Prompt-building helpers ----------


def test_format_context_numbers_chunks() -> None:
    chunks = [
        RetrievedChunk(
            text="alpha", doc_id="x", filename="a.pdf",
            page_number=1, chunk_index=0, score=0.9,
        ),
        RetrievedChunk(
            text="beta", doc_id="x", filename="a.pdf",
            page_number=2, chunk_index=1, score=0.5,
        ),
    ]
    rendered = _format_context(chunks)
    assert "[1]" in rendered and "[2]" in rendered
    assert "alpha" in rendered and "beta" in rendered
    assert "page=1" in rendered and "page=2" in rendered


def test_build_user_message_includes_question_and_context() -> None:
    chunks = [
        RetrievedChunk(
            text="alpha", doc_id="x", filename="a.pdf",
            page_number=1, chunk_index=0, score=0.9,
        )
    ]
    msg = _build_user_message("q?", chunks)
    assert msg.startswith("Context:")
    assert "Question: q?" in msg


# ---------- Streaming ----------


def test_stream_yields_deltas_then_done(
    fake_provider: _FakeProvider, sample_pdf: Path
) -> None:
    ingest_document(sample_pdf)
    chunks = retrieve("hello world", top_k=2)

    events = list(generate_answer_stream("q?", chunks))
    deltas = [e.delta for e in events if not e.done]
    dones = [e for e in events if e.done]

    assert "".join(deltas) == "canned answer [1]"
    assert len(dones) == 1
    assert dones[0].model == "fake-model"
    assert dones[0].input_tokens == 42
    assert dones[0].output_tokens == 7
    # Provider was called with the new question.
    assert fake_provider.last_messages is not None
    assert "q?" in fake_provider.last_messages[-1]["content"]


def test_stream_empty_chunks_skips_provider(fake_provider: _FakeProvider) -> None:
    events = list(generate_answer_stream("anything", chunks=[]))
    assert "".join(e.delta for e in events if not e.done) == NO_CONTEXT_ANSWER
    assert events[-1].done is True
    assert events[-1].input_tokens == 0
    assert fake_provider.last_messages is None  # provider untouched


def test_stream_empty_question_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        list(generate_answer_stream("   ", chunks=[]))


def test_build_messages_appends_new_question_last() -> None:
    chunks = [
        RetrievedChunk(
            text="t", doc_id="x", filename="a.pdf",
            page_number=1, chunk_index=0, score=0.5,
        )
    ]
    msgs = _build_messages(
        "current",
        chunks,
        history=[{"role": "user", "content": "prev"}],
    )
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "prev"}
    assert msgs[-1]["role"] == "user"
    assert "current" in msgs[-1]["content"]
