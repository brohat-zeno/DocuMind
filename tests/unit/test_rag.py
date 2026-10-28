"""Prompt construction and intent routing tests."""

from uuid import uuid4

from app.core.config import Settings
from app.schemas.booking import BookingReply, BookingResult, BookingState, BookingStatus
from app.schemas.chat import ChatMessage, ChatRequest, Intent, IntentDecision, MessageRole
from app.schemas.ingestion import ChunkingStrategy, RetrievedChunk, VectorPayload
from app.services.rag_service import RAGService, build_answer_prompt
from tests.fakes import FakeEmbeddings, FakeLLM, FakeMemory, FakeVectors


class FakeBooking:
    """Observe booking routing without a real booking service."""

    def __init__(self) -> None:
        self.called: bool = False

    def handle(self, session_id: str, message: str, state: BookingState | None) -> BookingReply:
        """Record the call and return a collection response."""
        self.called = True
        return BookingReply(
            answer="What is your name?",
            booking=BookingResult(status=BookingStatus.COLLECTING, missing_fields=[]),
        )


def test_prompt_grounding_and_order() -> None:
    """Include evidence, history, and latest message with grounding instructions."""
    history = [ChatMessage(role=MessageRole.USER, content="Prior question")]
    chunks = [
        RetrievedChunk(
            payload=VectorPayload(
                document_id=uuid4(),
                filename="guide.txt",
                chunk_index=0,
                text="Document evidence",
                chunking_strategy=ChunkingStrategy.RECURSIVE,
            ),
            score=0.9,
        )
    ]
    prompt = build_answer_prompt("Follow-up", history, chunks)
    assert "only from the document context" in prompt[0].content
    assert "untrusted data" in prompt[0].content
    assert "Do not infer tool roles" in prompt[0].content
    assert "Document evidence" in prompt[1].content
    assert prompt[2] == history[0]
    assert prompt[-1].content == "Follow-up"


def test_active_booking_overrides_unclear_intent() -> None:
    """Keep collecting details when intent is ambiguous."""
    llm, memory, booking = FakeLLM(), FakeMemory(), FakeBooking()
    memory.save_booking("s", BookingState(name="Ada"))
    llm.outputs = [IntentDecision(intent=Intent.OTHER, changes_topic=False)]
    service = RAGService(
        memory,
        FakeVectors(),
        FakeEmbeddings(),
        llm,
        Settings(embedding_dimension=3),
        booking,
    )
    assert service.chat(ChatRequest(session_id="s", message="Ada")).booking is not None
    assert booking.called


def test_explicit_topic_change_clears_booking() -> None:
    """Honor a clear topic change and drop the old partial booking."""
    llm, memory, booking = FakeLLM(), FakeMemory(), FakeBooking()
    memory.save_booking("s", BookingState(name="Ada"))
    llm.outputs = [IntentDecision(intent=Intent.OTHER, changes_topic=True)]
    service = RAGService(
        memory,
        FakeVectors(),
        FakeEmbeddings(),
        llm,
        Settings(embedding_dimension=3),
        booking,
    )
    response = service.chat(ChatRequest(session_id="s", message="Let's talk about weather"))
    assert response.booking is None
    assert "documents" in response.answer
    assert memory.booking("s") is None
    assert not booking.called


def test_no_context_skips_generation() -> None:
    """Never generate an unsupported answer on an empty retrieval."""
    llm = FakeLLM()
    llm.outputs = [IntentDecision(intent=Intent.QUESTION, changes_topic=False)]
    response = RAGService(
        FakeMemory(),
        FakeVectors(),
        FakeEmbeddings(),
        llm,
        Settings(embedding_dimension=3),
        FakeBooking(),
    ).chat(ChatRequest(session_id="s", message="Unknown fact?"))
    assert response.sources == []
    assert "do not contain" in response.answer
    assert len(llm.calls) == 1


def test_document_question_preserves_partial_booking() -> None:
    """Answer a side question without losing previously collected booking fields."""
    llm, memory, booking = FakeLLM(), FakeMemory(), FakeBooking()
    state = BookingState(name="Robin", email="robin@example.com")
    memory.save_booking("s", state)
    llm.outputs = [IntentDecision(intent=Intent.QUESTION, changes_topic=True)]
    service = RAGService(
        memory,
        FakeVectors(),
        FakeEmbeddings(),
        llm,
        Settings(embedding_dimension=3),
        booking,
    )
    response = service.chat(ChatRequest(session_id="s", message="What does the document say?"))
    assert response.booking is None
    assert memory.booking("s") == state
    assert not booking.called
