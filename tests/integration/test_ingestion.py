"""HTTP ingestion against real SQLite and in-memory external dependencies."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api.deps import get_ingestion_service
from app.core.config import Settings
from app.db.session import create_database
from app.main import create_app
from app.models.base import Base
from app.models.document import Document
from app.repositories.document_repo import SQLDocumentRepository
from app.services.ingestion_service import IngestionService
from tests.fakes import FakeEmbeddings, FakeVectors
from tests.unit.test_extraction import pdf_bytes


@pytest.fixture
def ingestion() -> Iterator[tuple[TestClient, IngestionService, FakeVectors]]:
    """Provide a fully injected ingestion API with isolated SQLite."""
    settings = Settings(database_url="sqlite://", embedding_dimension=3, max_upload_bytes=2000)
    engine, sessions = create_database(settings.database_url)
    Base.metadata.create_all(engine)
    vectors = FakeVectors()
    service = IngestionService(SQLDocumentRepository(sessions), vectors, FakeEmbeddings(), settings)
    app = create_app(settings)
    app.dependency_overrides[get_ingestion_service] = lambda: service
    with TestClient(app) as client:
        yield client, service, vectors
    engine.dispose()


@pytest.mark.parametrize("strategy", ["fixed_size", "recursive"])
@pytest.mark.parametrize(
    "filename,mime,content",
    [
        ("a.txt", "text/plain", b"First document. Second sentence."),
        ("a.pdf", "application/pdf", pdf_bytes("First document")),
    ],
)
def test_ingest_and_metadata(
    ingestion: tuple[TestClient, IngestionService, FakeVectors],
    strategy: str,
    filename: str,
    mime: str,
    content: bytes,
) -> None:
    """Persist an uploaded document and expose its metadata."""
    client, _, vectors = ingestion
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, content, mime)},
        data={"chunking_strategy": strategy, "chunk_size": 12, "chunk_overlap": 2},
    )
    assert response.status_code == 201, response.text
    result = response.json()
    metadata = client.get(f"/api/v1/documents/{result['document_id']}")
    assert metadata.status_code == 200
    assert metadata.json()["chunk_count"] == len(vectors.points)
    assert metadata.json()["size_bytes"] == len(content)
    assert all(point.payload.filename == filename for point in vectors.points)


@pytest.mark.parametrize(
    "filename,mime,content,data,status",
    [
        ("a.exe", "text/plain", b"x", {}, 400),
        ("a.txt", "application/pdf", b"x", {}, 400),
        ("a.txt", "text/plain", b"", {}, 400),
        ("a.txt", "text/plain", b"x" * 2001, {}, 413),
        ("a.pdf", "application/pdf", b"bad", {}, 400),
        ("a.txt", "text/plain", b"x", {"chunk_size": 4, "chunk_overlap": 4}, 422),
        ("a.txt", "text/plain", b"x", {"chunking_strategy": "invalid"}, 422),
    ],
)
def test_upload_errors(
    ingestion: tuple[TestClient, IngestionService, FakeVectors],
    filename: str,
    mime: str,
    content: bytes,
    data: dict[str, object],
    status: int,
) -> None:
    """Return useful errors without storing partial data."""
    client, _, vectors = ingestion
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, content, mime)},
        data={"chunking_strategy": "fixed_size", **data},
    )
    assert response.status_code == status
    assert response.json()
    assert vectors.points == []


def test_partial_vector_write_is_removed(
    ingestion: tuple[TestClient, IngestionService, FakeVectors],
) -> None:
    """Compensate even when upsert fails after a partial write."""
    client, _, vectors = ingestion
    vectors.fail = True
    response = client.post(
        "/api/v1/documents",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"chunking_strategy": "fixed_size"},
    )
    assert response.status_code == 503
    assert vectors.points == []


def test_sql_failure_after_commit_is_removed() -> None:
    """Delete both SQL metadata and vectors after an ambiguous SQL failure."""
    engine, sessions = create_database("sqlite://")
    Base.metadata.create_all(engine)
    documents = SQLDocumentRepository(sessions)
    original_save = documents.save
    vectors = FakeVectors()
    service = IngestionService(
        documents, vectors, FakeEmbeddings(), Settings(embedding_dimension=3)
    )
    from app.core.exceptions import DependencyError
    from app.schemas.ingestion import ChunkingStrategy, DocumentMetadata

    def fail_after_commit(document: DocumentMetadata) -> None:
        original_save(document)
        raise RuntimeError("commit acknowledgment lost")

    from io import BytesIO

    with patch.object(documents, "save", side_effect=fail_after_commit):
        with pytest.raises(DependencyError):
            service.ingest(
                "a.txt", "text/plain", BytesIO(b"hello"), ChunkingStrategy.FIXED_SIZE, None, None
            )
    assert vectors.points == []
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Document)) == 0
    engine.dispose()


def test_missing_metadata(ingestion: tuple[TestClient, IngestionService, FakeVectors]) -> None:
    """Return 404 for an unknown UUID."""
    client, _, _ = ingestion
    assert client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000").status_code == 404
