"""Generation service: build a grounded prompt from retrieved chunks and call an LLM.

The active LLM provider is selected by the LLM_PROVIDER env var (ollama | gemini |
anthropic). Each provider implements two tiny methods:

  • complete(system, messages) -> ProviderResponse        (non-streaming)
  • stream(system, messages)   -> Iterator[StreamEvent]   (token-by-token)

Providers are lazily imported so unused ones don't crash on missing creds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, TypedDict

import httpx

from backend.config import get_settings
from backend.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise document analyst. Answer the user's question using ONLY the "
    "numbered context chunks provided below. Rules:\n"
    "- Cite supporting chunks inline using [N] notation, where N is the chunk number "
    "shown in the context (e.g., [1], [2]).\n"
    "- If the chunks do not contain enough information to answer, say so explicitly. "
    "Do not invent facts or use outside knowledge.\n"
    "- Quote sparingly; prefer concise synthesis over copy-paste.\n"
    "- Keep answers under 200 words unless the question genuinely requires more detail.\n"
    "- When prior conversation turns are present, treat them as context for follow-up "
    "questions but ground every factual claim in the numbered chunks."
)

NO_CONTEXT_ANSWER = (
    "I couldn't find any relevant passages in the indexed documents for this question. "
    "Try rephrasing, or ingest a document that covers this topic."
)

_MAX_OUTPUT_TOKENS = 1024

SUMMARY_SYSTEM_PROMPT = (
    "You are a concise document summarizer. Given the opening pages of a document, "
    "produce a 2-3 sentence TL;DR that captures (1) what the document is, (2) its "
    "subject or purpose, and (3) any key entity (person, company, project) it centers on. "
    "Reply with the summary only — no preamble, no markdown, no bullet points."
)


class Message(TypedDict):
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Citation:
    """A source attribution for a retrieved chunk used in an answer."""

    rank: int
    filename: str
    page_number: int
    chunk_index: int
    score: float


@dataclass(frozen=True)
class Answer:
    """A grounded answer plus the citations and usage metadata."""

    text: str
    citations: list[Citation]
    model: str
    input_tokens: int
    output_tokens: int


# ---------- Provider abstraction ----------


@dataclass(frozen=True)
class ProviderResponse:
    """A single completion result from any LLM provider."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class StreamEvent:
    """One unit emitted while streaming.

    Either a text delta (`delta` set, `done=False`) or a final terminator
    (`done=True`) carrying total usage. Providers always end the stream with
    exactly one `done=True` event.
    """

    delta: str = ""
    done: bool = False
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class _Provider(Protocol):
    def complete(self, system: str, messages: list[Message]) -> ProviderResponse: ...
    def stream(
        self, system: str, messages: list[Message]
    ) -> Iterator[StreamEvent]: ...


class _OllamaProvider:
    """Local Ollama via the native /api/chat endpoint. No SDK; uses httpx."""

    def _payload(self, system: str, messages: list[Message], stream: bool) -> dict:
        s = get_settings()
        return {
            "model": s.ollama_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
            "options": {"num_predict": _MAX_OUTPUT_TOKENS},
        }

    def _url(self) -> str:
        return f"{get_settings().ollama_base_url.rstrip('/')}/api/chat"

    def complete(self, system: str, messages: list[Message]) -> ProviderResponse:
        s = get_settings()
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(self._url(), json=self._payload(system, messages, stream=False))
                r.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(self._connect_hint()) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

        data = r.json()
        text = (data.get("message") or {}).get("content", "").strip()
        return ProviderResponse(
            text=text,
            model=data.get("model", s.ollama_model),
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
        )

    def stream(self, system: str, messages: list[Message]) -> Iterator[StreamEvent]:
        s = get_settings()
        model = s.ollama_model
        in_tok = out_tok = 0
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream(
                    "POST", self._url(), json=self._payload(system, messages, stream=True)
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        obj = json.loads(line)
                        model = obj.get("model", model)
                        msg = obj.get("message") or {}
                        delta = msg.get("content") or ""
                        if delta:
                            yield StreamEvent(delta=delta)
                        if obj.get("done"):
                            in_tok = int(obj.get("prompt_eval_count", 0) or 0)
                            out_tok = int(obj.get("eval_count", 0) or 0)
        except httpx.ConnectError as exc:
            raise RuntimeError(self._connect_hint()) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama returned {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        yield StreamEvent(done=True, model=model, input_tokens=in_tok, output_tokens=out_tok)

    def _connect_hint(self) -> str:
        s = get_settings()
        return (
            f"Could not reach Ollama at {s.ollama_base_url}. "
            f"Is `ollama serve` running and have you pulled `{s.ollama_model}` "
            f"with `ollama pull {s.ollama_model}`?"
        )


class _GeminiProvider:
    """Google Gemini via google-generativeai SDK. Multi-turn via ChatSession."""

    def _start_chat(self, system: str, messages: list[Message]):
        import google.generativeai as genai  # lazy import

        s = get_settings()
        if not s.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env or switch LLM_PROVIDER."
            )
        if not messages:
            raise RuntimeError("Gemini provider received empty messages list.")
        genai.configure(api_key=s.gemini_api_key)
        model = genai.GenerativeModel(model_name=s.gemini_model, system_instruction=system)
        # Replay all but the last message as chat history; send the last as the new turn.
        history = [
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [m["content"]],
            }
            for m in messages[:-1]
        ]
        chat = model.start_chat(history=history)
        gen_config = genai.types.GenerationConfig(max_output_tokens=_MAX_OUTPUT_TOKENS)
        return chat, messages[-1]["content"], gen_config

    def complete(self, system: str, messages: list[Message]) -> ProviderResponse:
        s = get_settings()
        chat, last, gen_config = self._start_chat(system, messages)
        response = chat.send_message(last, generation_config=gen_config)
        text = response.text.strip() if response.text else ""

        input_tokens = output_tokens = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

        return ProviderResponse(
            text=text,
            model=s.gemini_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def stream(self, system: str, messages: list[Message]) -> Iterator[StreamEvent]:
        s = get_settings()
        chat, last, gen_config = self._start_chat(system, messages)
        response = chat.send_message(last, generation_config=gen_config, stream=True)
        for chunk in response:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield StreamEvent(delta=text)
        # Resolve the iterator so usage_metadata is populated.
        try:
            response.resolve()
        except Exception:  # pragma: no cover — older SDK shapes
            pass
        in_tok = out_tok = 0
        usage = getattr(response, "usage_metadata", None)
        if usage:
            in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        yield StreamEvent(
            done=True, model=s.gemini_model, input_tokens=in_tok, output_tokens=out_tok
        )


class _AnthropicProvider:
    """Anthropic Claude via the anthropic SDK. Native streaming + multi-turn."""

    def _client_and_args(self, system: str, messages: list[Message]):
        from anthropic import Anthropic  # lazy import

        s = get_settings()
        if not s.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env or switch LLM_PROVIDER."
            )
        client = Anthropic(api_key=s.anthropic_api_key)
        kwargs = dict(
            model=s.anthropic_model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        return client, kwargs

    def complete(self, system: str, messages: list[Message]) -> ProviderResponse:
        client, kwargs = self._client_and_args(system, messages)
        response = client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return ProviderResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def stream(self, system: str, messages: list[Message]) -> Iterator[StreamEvent]:
        client, kwargs = self._client_and_args(system, messages)
        with client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                if text:
                    yield StreamEvent(delta=text)
            final = s.get_final_message()
        yield StreamEvent(
            done=True,
            model=final.model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )


@lru_cache(maxsize=1)
def _get_provider() -> _Provider:
    """Pick the active provider based on LLM_PROVIDER. Cached for the process lifetime."""
    name = get_settings().llm_provider
    logger.info("Using LLM provider: %s", name)
    if name == "ollama":
        return _OllamaProvider()
    if name == "gemini":
        return _GeminiProvider()
    if name == "anthropic":
        return _AnthropicProvider()
    raise RuntimeError(f"Unknown LLM_PROVIDER: {name!r}")


# ---------- Prompt building (provider-agnostic) ----------


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered context block for the LLM."""
    blocks: list[str] = []
    for rank, chunk in enumerate(chunks, start=1):
        header = (
            f"[{rank}] source={chunk.filename} page={chunk.page_number} "
            f"chunk={chunk.chunk_index} score={chunk.score:.3f}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def _build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the user-turn content: context block + question."""
    context = _format_context(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def _chunks_to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            rank=i,
            filename=c.filename,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            score=c.score,
        )
        for i, c in enumerate(chunks, start=1)
    ]


def _build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[Message] | None,
) -> list[Message]:
    """Compose the full message list to send to the provider.

    Prior turns are replayed as plain text (their original retrieved chunks are not
    re-attached). The newest question carries the freshly retrieved chunks.
    """
    msgs: list[Message] = []
    if history:
        for turn in history:
            if turn["role"] not in ("user", "assistant") or not turn["content"]:
                continue
            msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append({"role": "user", "content": _build_user_message(question, chunks)})
    return msgs


def _model_label_for_no_context() -> str:
    s = get_settings()
    return {
        "ollama": s.ollama_model,
        "gemini": s.gemini_model,
        "anthropic": s.anthropic_model,
    }.get(s.llm_provider, s.llm_provider)


# ---------- Public API ----------


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[Message] | None = None,
) -> Answer:
    """Build a grounded answer from retrieved chunks via the active LLM provider.

    Args:
        question: The new natural-language question.
        chunks: Retrieved context chunks for `question`.
        history: Optional prior conversation turns (alternating user/assistant).
            Replayed before the new question; capped/sanitised by the caller.

    Returns:
        An Answer with the LLM's response, source citations built from `chunks`,
        and provider usage metadata. If `chunks` is empty, returns NO_CONTEXT_ANSWER
        without calling the provider.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    settings = get_settings()

    if not chunks:
        return Answer(
            text=NO_CONTEXT_ANSWER,
            citations=[],
            model=_model_label_for_no_context(),
            input_tokens=0,
            output_tokens=0,
        )

    provider = _get_provider()
    messages = _build_messages(question, chunks, history)
    result = provider.complete(SYSTEM_PROMPT, messages)

    logger.info(
        "Generated answer: provider=%s model=%s history_turns=%d in=%d out=%d",
        settings.llm_provider,
        result.model,
        len(history or []),
        result.input_tokens,
        result.output_tokens,
    )

    return Answer(
        text=result.text,
        citations=_chunks_to_citations(chunks),
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def summarize_text(text: str) -> str:
    """Generate a 2-3 sentence TL;DR of `text` via the active LLM provider.

    Returns an empty string if `text` is empty. Raises whatever the provider raises
    on transport errors — callers decide whether to swallow them (e.g. ingestion
    treats a summary failure as non-fatal).
    """
    if not text or not text.strip():
        return ""
    provider = _get_provider()
    messages: list[Message] = [{"role": "user", "content": text}]
    result = provider.complete(SUMMARY_SYSTEM_PROMPT, messages)
    return result.text.strip()


def generate_answer_stream(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[Message] | None = None,
) -> Iterator[StreamEvent]:
    """Streaming counterpart of `generate_answer`.

    Yields zero or more `StreamEvent(delta=...)` followed by exactly one terminal
    `StreamEvent(done=True, model=..., input_tokens=..., output_tokens=...)`.

    On empty `chunks`, the NO_CONTEXT_ANSWER is yielded as a single delta — the
    provider is never called — so callers can treat both branches uniformly.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    if not chunks:
        yield StreamEvent(delta=NO_CONTEXT_ANSWER)
        yield StreamEvent(
            done=True,
            model=_model_label_for_no_context(),
            input_tokens=0,
            output_tokens=0,
        )
        return

    provider = _get_provider()
    messages = _build_messages(question, chunks, history)
    saw_done = False
    for event in provider.stream(SYSTEM_PROMPT, messages):
        if event.done:
            saw_done = True
            logger.info(
                "Streamed answer: provider=%s model=%s history_turns=%d in=%d out=%d",
                get_settings().llm_provider,
                event.model,
                len(history or []),
                event.input_tokens,
                event.output_tokens,
            )
        yield event
    if not saw_done:
        # Defensive: providers should always terminate with done=True; synthesise one.
        yield StreamEvent(done=True, model=_model_label_for_no_context())
