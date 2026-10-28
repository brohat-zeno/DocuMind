"""HTTP multi-turn RAG with all external dependencies replaced."""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.deps import get_rag_service
from app.core.config import Settings
from app.main import create_app
from app.schemas.chat import Intent, IntentDecision, StandaloneQuery
from app.schemas.ingestion import ChunkingStrategy, VectorChunk, VectorPayload
from app.services.rag_service import RAGService
from tests.fakes import FakeEmbeddings, FakeLLM, FakeMemory, FakeVectors
from tests.unit.test_rag import FakeBooking


def test_multiturn_question_rewrites_and_retains_sources() -> None:
    """Resolve a follow-up query using the previous exchange."""
    llm, memory, vectors, embeddings = FakeLLM(), FakeMemory(), FakeVectors(), FakeEmbeddings()
    settings = Settings(database_url="sqlite://", embedding_dimension=3)
    document_id = uuid4()
    vectors.points = [
        VectorChunk(
            id=uuid4(),
            embedding=[1, 0, 0],
            payload=VectorPayload(
                document_id=document_id,
                filename="plans.txt",
                chunk_index=0,
                text="Plans: Basic and Plus. Plus includes support.",
                chunking_strategy=ChunkingStrategy.FIXED_SIZE,
            ),
        )
    ]
    llm.outputs = [
        IntentDecision(intent=Intent.QUESTION, changes_topic=False),
        IntentDecision(intent=Intent.QUESTION, changes_topic=False),
        StandaloneQuery(query="What does the Plus plan include?"),
    ]
    llm.answers = ["The plans are Basic and Plus.", "Plus includes support."]
    service = RAGService(memory, vectors, embeddings, llm, settings, FakeBooking())
    app = create_app(settings)
    app.dependency_overrides[get_rag_service] = lambda: service
    with TestClient(app) as client:
        first = client.post("/api/v1/chat", json={"session_id": "s", "message": "List the plans"})
        second = client.post(
            "/api/v1/chat", json={"session_id": "s", "message": "What about the second one?"}
        )
        assert first.status_code == second.status_code == 200
        assert second.json()["answer"] == "Plus includes support."
        assert second.json()["sources"] == [
            {
                "document_id": str(document_id),
                "filename": "plans.txt",
                "chunk_index": 0,
            }
        ]
        assert second.json()["booking"] is None
    assert embeddings.inputs == ["List the plans", "What does the Plus plan include?"]
    assert len(memory.history("s")) == 4
    assert memory.history("another-session") == []
    assert not llm.outputs


def test_chat_validation() -> None:
    """Reject whitespace-only identifiers and messages."""
    with TestClient(create_app(Settings(database_url="sqlite://"))) as client:
        for body in [{"session_id": "", "message": "x"}, {"session_id": "s", "message": "   "}]:
            assert client.post("/api/v1/chat", json=body).status_code == 422
