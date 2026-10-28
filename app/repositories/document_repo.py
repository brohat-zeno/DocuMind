"""Document persistence contracts and SQLite implementation."""

from typing import Protocol
from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from app.models.document import Document
from app.schemas.ingestion import DocumentMetadata


class DocumentRepository(Protocol):
    """Expose document metadata operations."""

    def save(self, document: DocumentMetadata) -> None:
        """Persist one document."""
        ...

    def get(self, document_id: UUID) -> DocumentMetadata | None:
        """Read a document if present."""
        ...

    def delete(self, document_id: UUID) -> None:
        """Remove a document if present."""
        ...


class SQLDocumentRepository:
    """Manage independent, short SQL transactions."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions: sessionmaker[Session] = sessions

    def save(self, document: DocumentMetadata) -> None:
        """Persist metadata in one transaction."""
        with self._sessions.begin() as session:
            session.add(Document(**document.model_dump()))

    def get(self, document_id: UUID) -> DocumentMetadata | None:
        """Return a detached metadata model."""
        with self._sessions() as session:
            row = session.get(Document, document_id)
            return DocumentMetadata.model_validate(row) if row else None

    def delete(self, document_id: UUID) -> None:
        """Delete metadata idempotently."""
        with self._sessions.begin() as session:
            session.execute(delete(Document).where(Document.id == document_id))
