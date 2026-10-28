"""Redis conversation memory with bounded history and expiration."""

from typing import Protocol

from redis import Redis

from app.core.config import Settings
from app.core.exceptions import DependencyError
from app.schemas.booking import BookingState
from app.schemas.chat import ChatMessage


class ChatMemoryRepository(Protocol):
    """Persist chat history and partial booking state."""

    def history(self, session_id: str) -> list[ChatMessage]:
        """Read the bounded conversation history."""
        ...

    def append(self, session_id: str, messages: list[ChatMessage]) -> None:
        """Append one exchange and refresh its expiration."""
        ...

    def booking(self, session_id: str) -> BookingState | None:
        """Read a partial booking."""
        ...

    def save_booking(self, session_id: str, state: BookingState) -> None:
        """Persist a partial booking with expiration."""
        ...

    def clear_booking(self, session_id: str) -> None:
        """Remove the partial booking."""
        ...


class RedisChatMemoryRepository:
    """Serialize typed conversation models into the required Redis keys."""

    def __init__(self, client: Redis, settings: Settings) -> None:
        self._client: Redis = client
        self._settings: Settings = settings

    def history(self, session_id: str) -> list[ChatMessage]:
        """Read only the latest configured number of exchanges."""
        try:
            values = self._client.lrange(
                f"chat:{session_id}:history", -2 * self._settings.history_turns, -1
            )
            return [ChatMessage.model_validate_json(value) for value in values]
        except Exception as exc:
            raise DependencyError("Conversation memory is unavailable.") from exc

    def append(self, session_id: str, messages: list[ChatMessage]) -> None:
        """Atomically append, trim, and expire the history list."""
        try:
            key = f"chat:{session_id}:history"
            with self._client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, *(message.model_dump_json() for message in messages))
                pipe.ltrim(key, -2 * self._settings.history_turns, -1)
                pipe.expire(key, self._settings.memory_ttl_seconds)
                pipe.execute()
        except Exception as exc:
            raise DependencyError("Conversation memory is unavailable.") from exc

    def booking(self, session_id: str) -> BookingState | None:
        """Deserialize partial state if the key has not expired."""
        try:
            value = self._client.get(f"chat:{session_id}:booking")
            return BookingState.model_validate_json(value) if value else None
        except Exception as exc:
            raise DependencyError("Booking memory is unavailable.") from exc

    def save_booking(self, session_id: str, state: BookingState) -> None:
        """Write a booking as JSON with the configured TTL."""
        try:
            self._client.set(
                f"chat:{session_id}:booking",
                state.model_dump_json(),
                ex=self._settings.memory_ttl_seconds,
            )
        except Exception as exc:
            raise DependencyError("Booking memory is unavailable.") from exc

    def clear_booking(self, session_id: str) -> None:
        """Delete partial state idempotently."""
        try:
            self._client.delete(f"chat:{session_id}:booking")
        except Exception as exc:
            raise DependencyError("Booking memory is unavailable.") from exc
