"""Document ingestion HTTP transport."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_ingestion_service
from app.schemas.ingestion import ChunkingStrategy, DocumentCreated, DocumentMetadata
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("", response_model=DocumentCreated, status_code=201)
def ingest_document(
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    file: Annotated[UploadFile, File()],
    chunking_strategy: Annotated[ChunkingStrategy, Form()],
    chunk_size: Annotated[int | None, Form(gt=0)] = None,
    chunk_overlap: Annotated[int | None, Form(ge=0)] = None,
) -> DocumentCreated:
    """Ingest a PDF or TXT upload."""
    return service.ingest(
        file.filename or "",
        file.content_type or "",
        file.file,
        chunking_strategy,
        chunk_size,
        chunk_overlap,
    )


@router.get("/{document_id}", response_model=DocumentMetadata)
def get_document(
    document_id: UUID,
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> DocumentMetadata:
    """Read metadata for a document."""
    return service.get(document_id)
