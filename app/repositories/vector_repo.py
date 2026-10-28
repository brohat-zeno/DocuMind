"""Qdrant vector persistence and search."""

from typing import Protocol
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.core.config import Settings
from app.core.exceptions import VectorStoreError
from app.schemas.ingestion import RetrievedChunk, VectorChunk, VectorPayload


class VectorRepository(Protocol):
    """Expose vector storage operations."""

    def upsert(self, chunks: list[VectorChunk]) -> None:
        """Store a batch of embedded chunks."""
        ...

    def delete_document(self, document_id: UUID) -> None:
        """Delete all chunks of a document."""
        ...

    def search(self, embedding: list[float], limit: int, threshold: float) -> list[RetrievedChunk]:
        """Retrieve similar document chunks."""
        ...


class QdrantVectorRepository:
    """Wrap the Qdrant client with typed data contracts."""

    def __init__(self, client: QdrantClient, settings: Settings) -> None:
        self._client: QdrantClient = client
        self._settings: Settings = settings

    def _ensure_collection(self) -> None:
        name = self._settings.qdrant_collection
        if not self._client.collection_exists(name):
            try:
                self._client.create_collection(
                    name,
                    vectors_config=models.VectorParams(
                        size=self._settings.embedding_dimension, distance=models.Distance.COSINE
                    ),
                )
            except Exception:
                # A concurrent ingestion may have created the same collection.
                if not self._client.collection_exists(name):
                    raise

        self._validate_dimension()

    def _validate_dimension(self) -> None:
        config = self._client.get_collection(self._settings.qdrant_collection).config.params.vectors
        if not isinstance(config, models.VectorParams):
            raise VectorStoreError("This collection must use a single unnamed dense vector.")
        if config.size != self._settings.embedding_dimension:
            raise VectorStoreError(
                f"Qdrant collection dimension is {config.size}, but the selected embedding "
                f"model requires {self._settings.embedding_dimension}. "
                "Use a new QDRANT_COLLECTION and re-ingest documents."
            )

    def upsert(self, chunks: list[VectorChunk]) -> None:
        """Create the collection if necessary and persist a batch."""
        try:
            self._ensure_collection()
            self._client.upsert(
                self._settings.qdrant_collection,
                points=[
                    models.PointStruct(
                        id=str(chunk.id),
                        vector=chunk.embedding,
                        payload=chunk.payload.model_dump(mode="json"),
                    )
                    for chunk in chunks
                ],
                wait=True,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Could not store document vectors.") from exc

    def delete_document(self, document_id: UUID) -> None:
        """Remove all document vectors using their payload identifier."""
        try:
            if self._client.collection_exists(self._settings.qdrant_collection):
                self._client.delete(
                    self._settings.qdrant_collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_id",
                                    match=models.MatchValue(value=str(document_id)),
                                )
                            ]
                        )
                    ),
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreError("Could not clean up document vectors.") from exc

    def search(self, embedding: list[float], limit: int, threshold: float) -> list[RetrievedChunk]:
        """Return validated payloads above the score threshold."""
        try:
            if not self._client.collection_exists(self._settings.qdrant_collection):
                return []
            self._validate_dimension()
            response = self._client.query_points(
                self._settings.qdrant_collection,
                query=embedding,
                limit=limit,
                score_threshold=threshold,
                with_payload=True,
            )
            return [
                RetrievedChunk(
                    payload=VectorPayload.model_validate(point.payload), score=point.score
                )
                for point in response.points
            ]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Could not retrieve document vectors.") from exc
