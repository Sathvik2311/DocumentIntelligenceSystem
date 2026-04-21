"""FastAPI application entry point.

Run locally:
    uvicorn backend.main:app --reload
"""

import logging

from fastapi import FastAPI

from backend.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Intelligence System",
    version="0.1.0",
)

# TODO: Week 2 — wire routers once implemented.
# from backend.routers import documents, query
# app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
# app.include_router(query.router, prefix="/api", tags=["query"])


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
