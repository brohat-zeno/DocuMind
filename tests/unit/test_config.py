"""Configuration checks."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_positive_upload_limit() -> None:
    """Reject unusable upload limits."""
    with pytest.raises(ValidationError):
        Settings(max_upload_bytes=0)


@pytest.mark.parametrize("timezone", ["Not/AZone", "../UTC"])
def test_invalid_booking_timezone(timezone: str) -> None:
    """Reject timezones that cannot resolve calendar dates."""
    with pytest.raises(ValidationError):
        Settings(booking_timezone=timezone)


def test_invalid_chunk_defaults() -> None:
    """Fail startup for contradictory default chunk settings."""
    with pytest.raises(ValidationError):
        Settings(chunk_size=10, chunk_overlap=10)
