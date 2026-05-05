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


class ConversationTurn(BaseModel):
    """One turn of prior conversation, replayed into the LLM for follow-ups."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question.")
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_id: str | None = Field(
        default=None,
        description="Restrict to a single document (legacy shortcut; prefer doc_ids).",
    )
    doc_ids: list[str] | None = Field(
        default=None,
        description=(
            "Restrict to a specific set of documents. Empty/null means search all. "
            "Takes precedence over doc_id when both are set."
        ),
    )
    filename: str | None = Field(default=None, description="Restrict to a single filename.")
    retrieval_only: bool = Field(
        default=False,
        description="If true, skip the LLM and return only the retrieved chunks.",
    )
    history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=20,
        description="Prior turns of the same conversation; replayed before the new question.",
    )
    use_hybrid: bool | None = Field(
        default=None,
        description="Override ENABLE_HYBRID_SEARCH. None = use server default.",
    )
    use_reranker: bool | None = Field(
        default=None,
        description="Override ENABLE_RERANKER. None = use server default.",
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
