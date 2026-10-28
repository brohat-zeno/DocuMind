"""Booking conversation data contracts."""

import re
from datetime import date, datetime, time
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class BookingStatus(StrEnum):
    """Booking conversation phases."""

    COLLECTING = "collecting"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class BookingField(StrEnum):
    """Required booking field names."""

    NAME = "name"
    EMAIL = "email"
    DATE = "date"
    TIME = "time"


class BookingState(BaseModel):
    """Represent a partially collected booking in Redis."""

    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None
    awaiting_confirmation: bool = False


class BookingResult(BaseModel):
    """Expose the current booking status to API clients."""

    status: BookingStatus
    missing_fields: list[BookingField]
    booking_id: UUID | None = None


class BookingReply(BaseModel):
    """Return booking conversational text and status."""

    answer: str
    booking: BookingResult


class BookingAction(StrEnum):
    """Actions extracted from the current booking message."""

    NONE = "none"
    CONFIRM = "confirm"
    CANCEL = "cancel"


class BookingExtraction(BaseModel):
    """Constrain LLM field updates and explicit conversational actions."""

    name: str | None
    email: str | None
    date: str | None
    time: str | None
    action: BookingAction


class BookingDetails(BaseModel):
    """Validate normalized booking details in application code."""

    name: str = Field(min_length=1)
    email: EmailStr
    booking_date: date = Field(alias="date")
    booking_time: time = Field(alias="time")

    @field_validator("name", "email", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        """Trim surrounding whitespace from identity fields."""
        return value.strip() if isinstance(value, str) else value

    @field_validator("booking_date", mode="before")
    @classmethod
    def require_iso_date(cls, value: object) -> object:
        """Accept only normalized ISO calendar dates."""
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Date must use YYYY-MM-DD.")
        return value

    @field_validator("booking_time", mode="before")
    @classmethod
    def require_clock_time(cls, value: object) -> object:
        """Accept only normalized minute-resolution 24-hour times."""
        if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}", value):
            raise ValueError("Time must use HH:MM.")
        return value


class BookingRecord(BaseModel):
    """Represent a confirmed SQL booking."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    email: str
    booking_date: date
    booking_time: time
    session_id: str
    created_at: datetime
