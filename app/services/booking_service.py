"""LLM-assisted collection with deterministic validation and persistence."""

import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import SlotUnavailableError
from app.repositories.booking_repo import BookingRepository
from app.repositories.chat_memory_repo import ChatMemoryRepository
from app.schemas.booking import (
    BookingAction,
    BookingDetails,
    BookingExtraction,
    BookingField,
    BookingRecord,
    BookingReply,
    BookingResult,
    BookingState,
    BookingStatus,
)
from app.schemas.chat import ChatMessage, MessageRole
from app.services.llm import LLMProvider


def merge_booking(state: BookingState, extraction: BookingExtraction) -> BookingState:
    """Merge only explicitly supplied fields and invalidate changed confirmations."""
    updates = extraction.model_dump(exclude={"action"}, exclude_none=True)
    merged = BookingState.model_validate({**state.model_dump(), **updates})
    if any(getattr(state, field) != getattr(merged, field) for field in BookingField):
        merged.awaiting_confirmation = False
    return merged


def validate_booking(
    state: BookingState, today: date
) -> tuple[BookingDetails | None, list[BookingField]]:
    """Identify every missing or invalid field, including past calendar dates."""
    invalid: set[BookingField] = set()
    details: BookingDetails | None = None
    try:
        details = BookingDetails.model_validate(state.model_dump())
    except ValidationError as exc:
        for error in exc.errors():
            invalid.add(BookingField(str(error["loc"][0])))
    # Check date separately so a valid-but-past date is reported alongside invalid email/time.
    if state.date:
        try:
            if date.fromisoformat(state.date) <= today:
                invalid.add(BookingField.DATE)
        except ValueError:
            invalid.add(BookingField.DATE)
    missing = [field for field in BookingField if field in invalid]
    return (None, missing) if missing else (details, [])


class BookingService:
    """Collect, validate, confirm, and reserve interview slots."""

    def __init__(
        self,
        bookings: BookingRepository,
        memory: ChatMemoryRepository,
        llm: LLMProvider,
        settings: Settings,
        today: Callable[[], date] | None = None,
    ) -> None:
        self._bookings: BookingRepository = bookings
        self._memory: ChatMemoryRepository = memory
        self._llm: LLMProvider = llm
        self._settings: Settings = settings
        self._today: Callable[[], date] = today or (
            lambda: datetime.now(ZoneInfo(settings.booking_timezone)).date()
        )

    def handle(self, session_id: str, message: str, state: BookingState | None) -> BookingReply:
        """Process extraction, validation, confirmation, and persistence in order."""
        previous = state or BookingState()
        today = self._today()
        extraction = self._llm.structured(
            [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "Extract interview booking updates from the latest message. "
                        "Return null for fields not supplied or corrected in the latest message. "
                        "Never invent names, emails, dates, or times. Normalize natural dates to "
                        "YYYY-MM-DD and times to HH:MM in 24h format. Keep invalid supplied text "
                        "for code validation. action is confirm only for an explicit affirmative "
                        "confirmation, cancel for explicit cancellation or a no to the final "
                        "confirmation, and none otherwise. A field correction is not cancellation. "
                        f"Today is {today.isoformat()} in {self._settings.booking_timezone}. "
                        f"Partial booking: {previous.model_dump_json()}"
                    ),
                ),
                ChatMessage(role=MessageRole.USER, content=message),
            ],
            BookingExtraction,
        )
        if extraction.action == BookingAction.CANCEL:
            self._memory.clear_booking(session_id)
            return BookingReply(
                answer="The interview booking has been cancelled.",
                booking=BookingResult(status=BookingStatus.CANCELLED, missing_fields=[]),
            )
        merged = merge_booking(previous, extraction)
        details, missing = validate_booking(merged, today)
        if missing:
            merged.awaiting_confirmation = False
            self._memory.save_booking(session_id, merged)
            return self._collecting(missing)
        assert details is not None
        if self._bookings.slot_taken(details.booking_date, details.booking_time):
            return self._slot_conflict(session_id, merged)
        explicit_yes = re.sub(r"[^\w\s]", "", message.lower()).strip()
        explicit_yes = " ".join(explicit_yes.split())
        confirmed = (
            previous.awaiting_confirmation
            and merged.awaiting_confirmation
            and extraction.action == BookingAction.CONFIRM
            and explicit_yes in self._settings.booking_confirmation_phrases
        )
        if confirmed:
            return self._persist(session_id, details, merged)
        merged.awaiting_confirmation = True
        self._memory.save_booking(session_id, merged)
        return BookingReply(
            answer=(
                f"Please confirm: {details.name}, {details.email}, "
                f"{details.booking_date.isoformat()} at {details.booking_time.strftime('%H:%M')} "
                f"({self._settings.booking_timezone}). Reply yes to book or no to cancel."
            ),
            booking=BookingResult(status=BookingStatus.AWAITING_CONFIRMATION, missing_fields=[]),
        )

    def _collecting(self, missing: list[BookingField]) -> BookingReply:
        return BookingReply(
            answer=f"Please provide a valid {', '.join(missing)}.",
            booking=BookingResult(status=BookingStatus.COLLECTING, missing_fields=missing),
        )

    def _slot_conflict(self, session_id: str, state: BookingState) -> BookingReply:
        state.time = None
        state.awaiting_confirmation = False
        self._memory.save_booking(session_id, state)
        return BookingReply(
            answer="That date and time are already booked. Please choose a different time or date.",
            booking=BookingResult(
                status=BookingStatus.COLLECTING, missing_fields=[BookingField.TIME]
            ),
        )

    def _persist(
        self,
        session_id: str,
        details: BookingDetails,
        state: BookingState,
    ) -> BookingReply:
        booking = BookingRecord(
            id=uuid4(),
            name=details.name,
            email=str(details.email),
            booking_date=details.booking_date,
            booking_time=details.booking_time,
            session_id=session_id,
            created_at=datetime.now(UTC),
        )
        try:
            self._bookings.save(booking)
        except SlotUnavailableError:
            return self._slot_conflict(session_id, state)
        self._memory.clear_booking(session_id)
        return BookingReply(
            answer=f"Your interview is confirmed. Booking id: {booking.id}.",
            booking=BookingResult(
                status=BookingStatus.CONFIRMED,
                missing_fields=[],
                booking_id=booking.id,
            ),
        )
