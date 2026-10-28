"""Domain errors and their HTTP mapping."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class DomainError(Exception):
    """Describe an expected application failure."""

    status_code: int = 400
    code: str = "invalid_request"


class UnsupportedFileTypeError(DomainError):
    """Indicate an unsupported upload format."""

    code = "unsupported_file_type"


class ExtractionError(DomainError):
    """Indicate unreadable or empty document text."""

    code = "extraction_error"


class UploadTooLargeError(DomainError):
    """Indicate an upload beyond the configured limit."""

    status_code = 413
    code = "upload_too_large"


class InvalidChunkingError(DomainError):
    """Indicate invalid chunking parameters."""

    status_code = 422
    code = "invalid_chunking"


class NotFoundError(DomainError):
    """Indicate missing persisted data."""

    status_code = 404
    code = "not_found"


class DependencyError(DomainError):
    """Indicate an unavailable dependency."""

    status_code = 503
    code = "dependency_error"


class VectorStoreError(DependencyError):
    """Indicate a vector database operation failed."""

    code = "vector_store_error"


class SlotUnavailableError(DomainError):
    """Indicate an already booked slot."""

    status_code = 409
    code = "slot_unavailable"


def register_handlers(app: FastAPI) -> None:
    """Register consistent domain and unexpected error responses."""

    @app.exception_handler(DomainError)
    async def domain_handler(request: Request, exc: DomainError) -> JSONResponse:
        logger.warning("Request failed path=%s code=%s", request.url.path, exc.code)
        return JSONResponse(
            status_code=exc.status_code, content={"error": {"code": exc.code, "message": str(exc)}}
        )

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
        # Avoid exception text: provider errors can contain document text or secrets.
        logger.error("Unexpected failure path=%s type=%s", request.url.path, type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "An internal error occurred."}},
        )
