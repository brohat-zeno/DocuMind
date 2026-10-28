"""Document ingestion orchestration with compensating cleanup."""

import logging
from datetime import UTC, datetime
from typing import BinaryIO
from uuid import UUID, uuid4, uuid5

from app.core.config import Settings
from app.core.exceptions import DependencyError, NotFoundError
from app.repositories.document_repo import DocumentRepository
from app.repositories.vector_repo import VectorRepository
from app.schemas.ingestion import (
    ChunkingStrategy,
    DocumentCreated,
    DocumentMetadata,
    VectorChunk,
    VectorPayload,
)
from app.services.chunking import Chunker, FixedSizeChunker, RecursiveChunker, validate_chunking
from app.services.embedding import EmbeddingProvider
from app.services.text_extraction import extract_text

logger = logging.getLogger(__name__)


class IngestionService:
    """Validate, extract, embed, and persist uploaded documents."""

    def __init__(
        self,
        documents: DocumentRepository,
        vectors: VectorRepository,
        embeddings: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self._documents: DocumentRepository = documents
        self._vectors: VectorRepository = vectors
        self._embeddings: EmbeddingProvider = embeddings
        self._settings: Settings = settings

    def ingest(
        self,
        filename: str,
        content_type: str,
        stream: BinaryIO,
        strategy: ChunkingStrategy,
        size: int | None,
        overlap: int | None,
    ) -> DocumentCreated:
        """Ingest one upload and compensate for any partial write."""
        size = self._settings.chunk_size if size is None else size
        overlap = self._settings.chunk_overlap if overlap is None else overlap
        validate_chunking(size, overlap)
        content = stream.read(self._settings.max_upload_bytes + 1)
        text = extract_text(filename, content_type, content, self._settings.max_upload_bytes)
        chunker: Chunker = (
            FixedSizeChunker() if strategy == ChunkingStrategy.FIXED_SIZE else RecursiveChunker()
        )
        chunks = chunker.chunk(text, size, overlap)
        document_id = uuid4()
        metadata = DocumentMetadata(
            id=document_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            chunking_strategy=strategy,
            chunk_size=size,
            chunk_overlap=overlap,
            chunk_count=len(chunks),
            created_at=datetime.now(UTC),
        )
        write_attempted = False
        try:
            for start in range(0, len(chunks), self._settings.embedding_batch_size):
                batch = chunks[start : start + self._settings.embedding_batch_size]
                embeddings = self._embeddings.embed(batch)
                if len(embeddings) != len(batch) or any(
                    len(vector) != self._settings.embedding_dimension for vector in embeddings
                ):
                    raise DependencyError(
                        "The embedding provider returned invalid vector dimensions."
                    )
                points = [
                    VectorChunk(
                        id=uuid5(document_id, str(start + index)),
                        embedding=embedding,
                        payload=VectorPayload(
                            document_id=document_id,
                            chunk_index=start + index,
                            text=chunk,
                            filename=filename,
                            chunking_strategy=strategy,
                        ),
                    )
                    for index, (chunk, embedding) in enumerate(zip(batch, embeddings, strict=True))
                ]
                write_attempted = True
                self._vectors.upsert(points)
            self._documents.save(metadata)
        except Exception as exc:
            if write_attempted:
                self._rollback(document_id)
            if isinstance(exc, DependencyError):
                raise
            raise DependencyError(
                "Document ingestion failed; partial writes were removed."
            ) from exc
        logger.info("Ingested document id=%s chunks=%d", document_id, len(chunks))
        return DocumentCreated(
            document_id=document_id,
            filename=filename,
            chunking_strategy=strategy,
            chunk_count=len(chunks),
        )

    def _rollback(self, document_id: UUID) -> None:
        failed = False
        for cleanup in (self._vectors.delete_document, self._documents.delete):
            try:
                cleanup(document_id)
            except Exception:
                failed = True
                logger.error("Ingestion cleanup failed document_id=%s", document_id)
        if failed:
            raise DependencyError(
                "Ingestion failed and storage is unavailable for cleanup; "
                "retry cleanup is required."
            )

    def get(self, document_id: UUID) -> DocumentMetadata:
        """Return stored metadata or raise a missing-document error."""
        document = self._documents.get(document_id)
        if document is None:
            raise NotFoundError("Document not found.")
        return document
