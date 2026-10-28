"""Complete booking conversations and SQL slot conflicts."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_rag_service
from app.core.config import Settings
from app.core.exceptions import SlotUnavailableError
from app.db.session import create_database
from app.main import create_app
from app.models.base import Base
from app.models.booking import Booking
from app.repositories.booking_repo import SQLBookingRepository
from app.schemas.booking import BookingAction, BookingExtraction
from app.schemas.chat import Intent, IntentDecision
from app.services.booking_service import BookingService
from app.services.rag_service import RAGService
from tests.fakes import FakeEmbeddings, FakeLLM, FakeMemory, FakeVectors
from tests.unit.test_booking import extraction


@dataclass
class BookingHarness:
    """Bundle injected dependencies for booking conversations."""

    client: TestClient
    llm: FakeLLM
    memory: FakeMemory
    sessions: sessionmaker[Session]
    repository: SQLBookingRepository

    def turn(
        self, message: str, output: BookingExtraction, session: str = "s"
    ) -> dict[str, object]:
        """Send a booking message with queued classification and extraction."""
        self.llm.outputs.extend(
            [
                IntentDecision(intent=Intent.BOOKING, changes_topic=False),
                output,
            ]
        )
        response = self.client.post(
            "/api/v1/chat", json={"session_id": session, "message": message}
        )
        assert response.status_code == 200, response.text
        return response.json()

    def count(self) -> int:
        """Count persisted bookings."""
        with self.sessions() as session:
            return session.scalar(select(func.count()).select_from(Booking)) or 0


@pytest.fixture
def booking() -> Iterator[BookingHarness]:
    """Use actual SQLite with fake LLM, vector, embedding, and Redis dependencies."""
    settings = Settings(database_url="sqlite://", embedding_dimension=3)
    engine, sessions = create_database("sqlite://")
    Base.metadata.create_all(engine)
    llm, memory = FakeLLM(), FakeMemory()
    repository = SQLBookingRepository(sessions)
    booking_service = BookingService(
        repository, memory, llm, settings, today=lambda: date(2030, 1, 1)
    )
    rag = RAGService(memory, FakeVectors(), FakeEmbeddings(), llm, settings, booking_service)
    app = create_app(settings)
    app.dependency_overrides[get_rag_service] = lambda: rag
    with TestClient(app) as client:
        yield BookingHarness(client, llm, memory, sessions, repository)
    engine.dispose()


def test_full_conversation(booking: BookingHarness) -> None:
    """Collect only missing fields, confirm explicitly, persist once, and clear state."""
    first = booking.turn("Book an interview", extraction())
    assert first["booking"] == {
        "status": "collecting",
        "missing_fields": ["name", "email", "date", "time"],
        "booking_id": None,
    }
    second = booking.turn("Ada, ada@example.com", extraction(name="Ada", email="ada@example.com"))
    assert second["booking"]["missing_fields"] == ["date", "time"]
    third = booking.turn("Tomorrow at 2pm", extraction(day="2030-01-02", time="14:00"))
    assert third["booking"]["status"] == "awaiting_confirmation"
    assert booking.count() == 0
    final = booking.turn("yes", extraction(action=BookingAction.CONFIRM))
    assert final["booking"]["status"] == "confirmed"
    assert final["booking"]["booking_id"] in final["answer"]
    assert booking.count() == 1
    assert booking.memory.booking("s") is None
    assert len(booking.memory.history("s")) == 8
    with booking.sessions() as session:
        row = session.scalar(select(Booking))
        assert row is not None
        assert row.name == "Ada"
        assert row.email == "ada@example.com"
        assert row.booking_date == date(2030, 1, 2)
        assert row.booking_time.hour == 14


def begin_complete_booking(booking: BookingHarness, session: str = "s") -> None:
    """Supply all required details and stop before confirmation."""
    response = booking.turn(
        "Ada, ada@example.com, tomorrow 2pm",
        extraction(name="Ada", email="ada@example.com", day="2030-01-02", time="14:00"),
        session,
    )
    assert response["booking"]["status"] == "awaiting_confirmation"


@pytest.mark.parametrize("message", ["no", "cancel the booking"])
def test_cancel(booking: BookingHarness, message: str) -> None:
    """Clear pending state on rejection or explicit cancellation."""
    begin_complete_booking(booking)
    result = booking.turn(message, extraction(action=BookingAction.CANCEL))
    assert result["booking"]["status"] == "cancelled"
    assert booking.memory.booking("s") is None
    assert booking.count() == 0


def test_correction_requires_new_confirmation(booking: BookingHarness) -> None:
    """Never confirm an updated slot using an old confirmation."""
    begin_complete_booking(booking)
    result = booking.turn("yes, but 3pm", extraction(time="15:00", action=BookingAction.CONFIRM))
    assert result["booking"]["status"] == "awaiting_confirmation"
    assert "15:00" in result["answer"]
    assert booking.count() == 0
    assert booking.turn("yes", extraction(action=BookingAction.CONFIRM))["booking"]["status"] == (
        "confirmed"
    )


def test_model_cannot_invent_explicit_confirmation(booking: BookingHarness) -> None:
    """Require an explicit affirmative in user text even if the LLM says confirm."""
    begin_complete_booking(booking)
    result = booking.turn("maybe later", extraction(action=BookingAction.CONFIRM))
    assert result["booking"]["status"] == "awaiting_confirmation"
    assert booking.count() == 0


def test_first_message_cannot_confirm(booking: BookingHarness) -> None:
    """Require a separate confirmation after presenting all validated details."""
    result = booking.turn(
        "yes",
        extraction(
            name="Ada",
            email="ada@example.com",
            day="2030-01-02",
            time="14:00",
            action=BookingAction.CONFIRM,
        ),
    )
    assert result["booking"]["status"] == "awaiting_confirmation"
    assert booking.count() == 0


def test_double_booking_and_new_slot(booking: BookingHarness) -> None:
    """Reject occupied slots and accept a different time after confirmation."""
    begin_complete_booking(booking)
    booking.turn("yes", extraction(action=BookingAction.CONFIRM))
    conflict = booking.turn(
        "Ada, ada@example.com, tomorrow 2pm",
        extraction(name="Ada", email="ada@example.com", day="2030-01-02", time="14:00"),
        "other",
    )
    assert conflict["booking"]["status"] == "collecting"
    assert "already booked" in conflict["answer"]
    replacement = booking.turn("3pm", extraction(time="15:00"), "other")
    assert replacement["booking"]["status"] == "awaiting_confirmation"
    booking.turn("yes", extraction(action=BookingAction.CONFIRM), "other")
    assert booking.count() == 2


def test_conflict_during_final_insert(booking: BookingHarness) -> None:
    """Recover when another request takes the slot after the availability check."""
    begin_complete_booking(booking)
    with patch.object(booking.repository, "save", side_effect=SlotUnavailableError("Taken")):
        response = booking.turn("yes", extraction(action=BookingAction.CONFIRM))
    assert response["booking"]["status"] == "collecting"
    assert "already booked" in response["answer"]
    assert not booking.memory.booking("s").awaiting_confirmation


def test_sql_uniqueness_is_enforced(booking: BookingHarness) -> None:
    """The database rejects conflicts independently of service availability checks."""
    from datetime import UTC, datetime, time
    from uuid import uuid4

    from app.schemas.booking import BookingRecord

    row = BookingRecord(
        id=uuid4(),
        name="Ada",
        email="ada@example.com",
        booking_date=date(2030, 1, 2),
        booking_time=time(14),
        session_id="s",
        created_at=datetime.now(UTC),
    )
    booking.repository.save(row)
    with pytest.raises(SlotUnavailableError):
        booking.repository.save(row.model_copy(update={"id": uuid4()}))
    assert booking.count() == 1


def test_invalid_fields_ask_only_for_invalid_values(booking: BookingHarness) -> None:
    """Keep valid values and ask only for invalid or missing ones."""
    response = booking.turn(
        "Ada, bad email, yesterday, 25:00",
        extraction(name="Ada", email="bad", day="2029-12-31", time="25:00"),
    )
    assert response["booking"]["missing_fields"] == ["email", "date", "time"]
    assert "name" not in response["answer"]


def test_booking_resumes_after_document_question(booking: BookingHarness) -> None:
    """Resume real booking validation and persistence after a question interrupts collection."""
    from app.schemas.chat import StandaloneQuery

    booking.turn("Book an interview. My name is Robin.", extraction(name="Robin"))
    before = booking.memory.booking("s")
    booking.llm.outputs.extend(
        [
            IntentDecision(intent=Intent.QUESTION, changes_topic=True),
            StandaloneQuery(query="What does the handbook say about training?"),
        ]
    )
    response = booking.client.post(
        "/api/v1/chat", json={"session_id": "s", "message": "What does the handbook say?"}
    )
    assert response.status_code == 200
    assert response.json()["booking"] is None
    assert booking.memory.booking("s") == before
    reply = booking.turn(
        "robin@example.com, tomorrow at 14:00",
        extraction(email="robin@example.com", day="2030-01-02", time="14:00"),
    )
    assert reply["booking"]["status"] == "awaiting_confirmation"
    assert "Robin" in reply["answer"]
    booking.turn("yes", extraction(action=BookingAction.CONFIRM))
    assert booking.count() == 1
    assert booking.memory.booking("s") is None
