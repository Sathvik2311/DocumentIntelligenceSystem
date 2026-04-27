"""Query endpoint: retrieve top-k chunks and generate a grounded answer."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from backend.models.schemas import CitationModel, QueryRequest, QueryResponse
from backend.services.generation import generate_answer
from backend.services.retrieval import retrieve

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse, summary="Ask a grounded question.")
async def query(req: QueryRequest) -> QueryResponse:
    try:
        chunks = retrieve(
            req.question,
            top_k=req.top_k,
            doc_id=req.doc_id,
            filename=req.filename,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    try:
        answer = generate_answer(req.question, chunks)
    except RuntimeError as exc:
        # e.g. ANTHROPIC_API_KEY not configured.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return QueryResponse(
        question=req.question,
        answer=answer.text,
        citations=[
            CitationModel(
                rank=c.rank,
                filename=c.filename,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                score=c.score,
            )
            for c in answer.citations
        ],
        model=answer.model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )
