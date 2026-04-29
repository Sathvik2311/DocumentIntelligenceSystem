"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- Documents ----------


class DocumentMetadata(BaseModel):
    """Aggregated view of one document in the corpus."""

    doc_id: str = Field(..., description="Server-assigned document identifier.")
    filename: str
    num_chunks: int = Field(..., ge=0)
    num_pages: int = Field(..., ge=0)


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]


class UploadResponse(BaseModel):
    """Returned after a successful upload + ingestion."""

    doc_id: str
    filename: str
    num_pages: int = Field(..., ge=0)
    num_chunks: int = Field(..., ge=0)


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int = Field(..., ge=0)


# ---------- Query ----------


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question.")
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_id: str | None = Field(default=None, description="Restrict to a single document.")
    filename: str | None = Field(default=None, description="Restrict to a single filename.")
    retrieval_only: bool = Field(
        default=False,
        description="If true, skip the LLM and return only the retrieved chunks.",
    )


class CitationModel(BaseModel):
    rank: int = Field(..., ge=1)
    filename: str
    page_number: int = Field(..., ge=0)
    chunk_index: int = Field(..., ge=0)
    score: float
    text: str | None = Field(
        default=None,
        description="Chunk text. Populated on retrieval_only or when explicitly requested.",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]
    model: str
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)


# ---------- Errors ----------


class ErrorResponse(BaseModel):
    """Structured error envelope returned for non-2xx responses."""

    status_code: int
    message: str
    detail: str | None = None
