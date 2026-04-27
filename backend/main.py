"""FastAPI application entry point.

Run locally:
    uvicorn backend.main:app --reload
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.models.schemas import ErrorResponse
from backend.routers import documents, query

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Intelligence System",
    version="0.1.0",
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(query.router, prefix="/api", tags=["query"])


@app.get("/api/health", tags=["health"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# ---------- Structured error responses ----------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    body = ErrorResponse(status_code=exc.status_code, message=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    body = ErrorResponse(
        status_code=422,
        message="Request validation failed.",
        detail=str(exc.errors()),
    )
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    body = ErrorResponse(status_code=500, message="Internal server error.", detail=str(exc))
    return JSONResponse(status_code=500, content=body.model_dump())
