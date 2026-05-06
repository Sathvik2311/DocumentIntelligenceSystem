"""Query endpoints: retrieve top-k chunks and generate a grounded answer.

Two flavors:
  • POST /query        — synchronous, returns the full answer at once (JSON).
  • POST /query/stream — Server-Sent Events: citations first, then token deltas,
                         then a terminal `done` event with usage metadata.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.models.schemas import CitationModel, QueryRequest, QueryResponse
from backend.services.generation import (
    NO_CONTEXT_ANSWER,
    generate_answer,
    generate_answer_stream,
)
from backend.services.retrieval import RetrievedChunk, retrieve

router = APIRouter()
logger = logging.getLogger(__name__)


RETRIEVAL_ONLY_ANSWER = "(retrieval-only mode — LLM not called)"


def _chunks_to_citations(chunks: list[RetrievedChunk], include_text: bool) -> list[CitationModel]:
    """Convert RetrievedChunks into API CitationModels, optionally with text bodies."""
    return [
        CitationModel(
            rank=i,
            filename=c.filename,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            score=c.score,
            text=c.text if include_text else None,
        )
        for i, c in enumerate(chunks, start=1)
    ]


def _retrieve_for_request(req: QueryRequest) -> list[RetrievedChunk]:
    try:
        return retrieve(
            req.question,
            top_k=req.top_k,
            doc_id=req.doc_id,
            doc_ids=req.doc_ids,
            filename=req.filename,
            use_hybrid=req.use_hybrid,
            use_reranker=req.use_reranker,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/query", response_model=QueryResponse, summary="Ask a grounded question.")
async def query(req: QueryRequest) -> QueryResponse:
    chunks = _retrieve_for_request(req)

    # Retrieval-only mode: skip the LLM, return chunk text in citations.
    if req.retrieval_only:
        return QueryResponse(
            question=req.question,
            answer=RETRIEVAL_ONLY_ANSWER if chunks else NO_CONTEXT_ANSWER,
            citations=_chunks_to_citations(chunks, include_text=True),
            model="(none)",
            input_tokens=0,
            output_tokens=0,
        )

    history = [{"role": t.role, "content": t.content} for t in req.history]

    try:
        answer = generate_answer(req.question, chunks, history=history)
    except RuntimeError as exc:
        # e.g. provider not configured / unreachable.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return QueryResponse(
        question=req.question,
        answer=answer.text,
        citations=_chunks_to_citations(chunks, include_text=True),
        model=answer.model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )


# ---------- Streaming ----------


def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/query/stream",
    summary="Ask a grounded question; stream tokens via Server-Sent Events.",
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """Stream the answer token-by-token.

    Event sequence:
      1. `citations` — full citation list (one event)
      2. `token`     — repeated, each carrying `{"text": "<delta>"}`
      3. `done`      — terminal event with `{model, input_tokens, output_tokens}`
      4. `error`     — only on provider failure (replaces `done`)

    `retrieval_only=true` short-circuits: emits citations + done with no tokens.
    """
    chunks = _retrieve_for_request(req)
    citations = _chunks_to_citations(chunks, include_text=True)
    history = [{"role": t.role, "content": t.content} for t in req.history]

    def event_gen() -> Iterator[str]:
        yield _sse("citations", {"citations": [c.model_dump() for c in citations]})

        if req.retrieval_only:
            yield _sse(
                "done",
                {"model": "(none)", "input_tokens": 0, "output_tokens": 0},
            )
            return

        try:
            for ev in generate_answer_stream(req.question, chunks, history=history):
                if ev.done:
                    yield _sse(
                        "done",
                        {
                            "model": ev.model,
                            "input_tokens": ev.input_tokens,
                            "output_tokens": ev.output_tokens,
                        },
                    )
                elif ev.delta:
                    yield _sse("token", {"text": ev.delta})
        except RuntimeError as exc:
            logger.warning("Streaming provider error: %s", exc)
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
