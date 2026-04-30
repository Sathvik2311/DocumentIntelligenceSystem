"""Query endpoint: retrieve top-k chunks and generate a grounded answer."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.models.schemas import CitationModel, QueryRequest, QueryResponse
from backend.services.generation import NO_CONTEXT_ANSWER, generate_answer
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


@router.post("/query", response_model=QueryResponse, summary="Ask a grounded question.")
async def query(req: QueryRequest) -> QueryResponse:
    try:
        chunks = retrieve(
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

    # Build citations from the retrieved chunks so the UI can display text on demand.
    # `answer.citations` already aligns 1:1 with `chunks` (same order, ranks 1..N).
    return QueryResponse(
        question=req.question,
        answer=answer.text,
        citations=_chunks_to_citations(chunks, include_text=True),
        model=answer.model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )
