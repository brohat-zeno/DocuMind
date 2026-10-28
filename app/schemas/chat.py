"""Chat transport and LLM structured output schemas."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.booking import BookingResult


class ChatRequest(BaseModel):
    """Validate a session message."""

    model_config = ConfigDict(str_strip_whitespace=True)
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class Source(BaseModel):
    """Identify a document chunk used as context."""

    document_id: UUID
    filename: str
    chunk_index: int


class ChatResponse(BaseModel):
    """Return a grounded answer or booking response."""

    session_id: str
    answer: str
    sources: list[Source]
    booking: BookingResult | None = None


class MessageRole(StrEnum):
    """Supported prompt and conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Represent a message across application layers."""

    role: MessageRole
    content: str


class Intent(StrEnum):
    """Supported conversational intents."""

    QUESTION = "question"
    BOOKING = "booking"
    OTHER = "other"


class IntentDecision(BaseModel):
    """Represent the LLM's classification and explicit topic change."""

    intent: Intent
    changes_topic: bool


class StandaloneQuery(BaseModel):
    """Represent a history-resolved retrieval query."""

    query: str = Field(min_length=1)
