"""Booking field normalization, validation, and merge behavior."""

from datetime import date

import pytest

from app.schemas.booking import BookingAction, BookingExtraction, BookingField, BookingState
from app.services.booking_service import merge_booking, validate_booking


def extraction(
    name: str | None = None,
    email: str | None = None,
    day: str | None = None,
    time: str | None = None,
    action: BookingAction = BookingAction.NONE,
) -> BookingExtraction:
    """Create a schema-conforming extraction fixture."""
    return BookingExtraction(name=name, email=email, date=day, time=time, action=action)


def test_merge_preserves_fields_and_requires_reconfirmation() -> None:
    """Updates must not erase collected fields or retain a stale confirmation."""
    state = BookingState(
        name="Ada",
        email="ada@example.com",
        date="2030-01-02",
        time="10:00",
        awaiting_confirmation=True,
    )
    merged = merge_booking(state, extraction(time="11:00"))
    assert merged.name == "Ada"
    assert merged.email == "ada@example.com"
    assert merged.time == "11:00"
    assert not merged.awaiting_confirmation
    assert state.time == "10:00"
    assert merge_booking(state, extraction()).awaiting_confirmation


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("name", "  "),
        ("email", "bad"),
        ("date", "2030-02-30"),
        ("date", "2020-01-01"),
        ("date", "2030-01-01"),
        ("date", "20300102"),
        ("time", "24:00"),
        ("time", "9:00"),
        ("time", "12:99"),
        ("time", "10:00:00"),
    ],
)
def test_invalid_fields(field: str, value: str) -> None:
    """Detect invalid values in code without trusting the model."""
    data = {"name": "Ada", "email": "ada@example.com", "date": "2030-01-02", "time": "10:00"}
    data[field] = value
    details, missing = validate_booking(BookingState.model_validate(data), date(2030, 1, 1))
    assert details is None
    assert missing == [BookingField(field)]


def test_all_missing_fields() -> None:
    """Ask for all required fields on an empty booking."""
    details, missing = validate_booking(BookingState(), date(2030, 1, 1))
    assert details is None
    assert missing == list(BookingField)


def test_multiple_invalid_fields() -> None:
    """Report a past date even when another field fails parsing."""
    _, missing = validate_booking(
        BookingState(name="Ada", email="bad", date="2020-01-01", time="25:00"),
        date(2030, 1, 1),
    )
    assert missing == [BookingField.EMAIL, BookingField.DATE, BookingField.TIME]


def test_valid_fields() -> None:
    """Normalize identity whitespace and parse date/time as typed values."""
    details, missing = validate_booking(
        BookingState(name=" Ada ", email="ada@example.com", date="2030-01-02", time="10:00"),
        date(2030, 1, 1),
    )
    assert not missing
    assert details is not None
    assert details.name == "Ada"
    assert details.booking_date == date(2030, 1, 2)
    assert details.booking_time.hour == 10
