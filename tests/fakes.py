"""In-memory dependency replacements for deterministic tests."""

from uuid import UUID

from pydantic import BaseModel

from app.core.exceptions import VectorStoreError
from app.schemas.booking import BookingState
from app.schemas.chat import ChatMessage
from app.schemas.ingestion import RetrievedChunk, VectorChunk
from app.services.llm import Output


class FakeEmbeddings:
    """Embed text deterministically without network access."""

    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.fail: bool = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Record inputs and return fixed-size vectors."""
        if self.fail:
            raise RuntimeError("Embedding failed")
        self.inputs.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectors:
    """Model persisted vectors including partial upsert failures."""

    def __init__(self) -> None:
        self.points: list[VectorChunk] = []
        self.fail: bool = False

    def upsert(self, chunks: list[VectorChunk]) -> None:
        """Save a batch and optionally fail after writing it."""
        self.points.extend(chunks)
        if self.fail:
            raise VectorStoreError("Simulated partial write")

    def delete_document(self, document_id: UUID) -> None:
        """Remove every matching document point."""
        self.points = [p for p in self.points if p.payload.document_id != document_id]

    def search(self, embedding: list[float], limit: int, threshold: float) -> list[RetrievedChunk]:
        """Return predictable scored chunks."""
        return [
            RetrievedChunk(payload=point.payload, score=0.9)
            for point in self.points[:limit]
            if 0.9 >= threshold
        ]


class FakeMemory:
    """Keep typed history and booking state in memory."""

    def __init__(self, turns: int = 10) -> None:
        self.histories: dict[str, list[ChatMessage]] = {}
        self.bookings: dict[str, BookingState] = {}
        self.turns = turns

    def history(self, session_id: str) -> list[ChatMessage]:
        """Return a copy of session history."""
        return list(self.histories.get(session_id, []))

    def append(self, session_id: str, messages: list[ChatMessage]) -> None:
        """Retain only the configured number of exchanges."""
        self.histories[session_id] = (self.history(session_id) + messages)[-2 * self.turns :]

    def booking(self, session_id: str) -> BookingState | None:
        """Return detached state."""
        value = self.bookings.get(session_id)
        return value.model_copy(deep=True) if value else None

    def save_booking(self, session_id: str, state: BookingState) -> None:
        """Store detached partial state."""
        self.bookings[session_id] = state.model_copy(deep=True)

    def clear_booking(self, session_id: str) -> None:
        """Delete a partial booking."""
        self.bookings.pop(session_id, None)


class FakeLLM:
    """Consume a strict queue of typed model responses."""

    def __init__(self) -> None:
        self.outputs: list[BaseModel] = []
        self.answers: list[str] = []
        self.calls: list[list[ChatMessage]] = []

    def structured(self, messages: list[ChatMessage], schema: type[Output]) -> Output:
        """Assert the expected schema and return a queued response."""
        self.calls.append(messages)
        output = self.outputs.pop(0)
        assert isinstance(output, schema), f"Expected {schema}, got {type(output)}"
        return schema.model_validate(output.model_dump())

    def complete(self, messages: list[ChatMessage]) -> str:
        """Return a queued plain-text answer."""
        self.calls.append(messages)
        return self.answers.pop(0)
