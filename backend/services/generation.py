"""Generation service: build a grounded prompt from retrieved chunks and call Anthropic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from anthropic import Anthropic

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
    "- Keep answers under 200 words unless the question genuinely requires more detail."
)

NO_CONTEXT_ANSWER = (
    "I couldn't find any relevant passages in the indexed documents for this question. "
    "Try rephrasing, or ingest a document that covers this topic."
)

_MAX_OUTPUT_TOKENS = 1024


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


@lru_cache(maxsize=1)
def _get_client() -> Anthropic:
    """Lazily build a single Anthropic client (uses ANTHROPIC_API_KEY from env)."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env before calling generate_answer()."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


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


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> Answer:
    """Call the Anthropic API with the question + retrieved chunks as context.

    If `chunks` is empty, returns a canned "no context" answer without calling the API.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")

    if not chunks:
        return Answer(
            text=NO_CONTEXT_ANSWER,
            citations=[],
            model=get_settings().anthropic_model,
            input_tokens=0,
            output_tokens=0,
        )

    settings = get_settings()
    client = _get_client()
    user_message = _build_user_message(question, chunks)

    # Cache the system prompt so repeat calls within ~5 min share it.
    # (When multi-turn conversation lands in Week 2, the per-document context
    # block will also become a cache target.)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=_MAX_OUTPUT_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    logger.info(
        "Generated answer: model=%s in=%d out=%d",
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    return Answer(
        text=text,
        citations=_chunks_to_citations(chunks),
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
