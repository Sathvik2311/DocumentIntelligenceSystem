"""Document management endpoints: upload, list, delete."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.models.schemas import (
    DeleteResponse,
    DocumentListResponse,
    DocumentMetadata,
    UploadResponse,
)
from backend.services.ingestion import (
    SUPPORTED_EXTENSIONS,
    delete_document,
    ingest_document,
    list_documents,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document.",
)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type {suffix!r}. Allowed: {sorted(SUPPORTED_EXTENSIONS)}.",
        )

    # Persist to a temp dir using the original filename so ingestion records
    # the right name in chunk metadata.
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / Path(file.filename).name
        with tmp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        try:
            result = ingest_document(tmp_path)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    if result.num_chunks == 0:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Document parsed to zero chunks (file may be empty or unreadable).",
        )

    return UploadResponse(
        doc_id=result.doc_id,
        filename=result.filename,
        num_pages=result.num_pages,
        num_chunks=result.num_chunks,
    )


@router.get("", response_model=DocumentListResponse, summary="List all ingested documents.")
async def list_all() -> DocumentListResponse:
    summaries = list_documents()
    return DocumentListResponse(
        documents=[
            DocumentMetadata(
                doc_id=s.doc_id,
                filename=s.filename,
                num_chunks=s.num_chunks,
                num_pages=s.num_pages,
            )
            for s in summaries
        ]
    )


@router.delete("/{document_id}", response_model=DeleteResponse, summary="Delete a document and its embeddings.")
async def delete(document_id: str) -> DeleteResponse:
    deleted = delete_document(document_id)
    if deleted == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No document with id {document_id!r}.")
    return DeleteResponse(doc_id=document_id, deleted_chunks=deleted)
