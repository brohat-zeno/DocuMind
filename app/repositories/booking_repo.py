"""Booking data access with atomic slot uniqueness."""

from datetime import date, time
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.exceptions import DependencyError, SlotUnavailableError
from app.models.booking import Booking
from app.schemas.booking import BookingRecord


class BookingRepository(Protocol):
    """Persist bookings and inspect slot occupancy."""

    def save(self, booking: BookingRecord) -> None:
        """Atomically reserve a slot or report a conflict."""
        ...

    def slot_taken(self, booking_date: date, booking_time: time) -> bool:
        """Report whether a confirmed booking occupies the slot."""
        ...


class SQLBookingRepository:
    """Protect slots through a database unique constraint."""

    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self._sessions: sessionmaker[Session] = sessions

    def save(self, booking: BookingRecord) -> None:
        """Insert one booking in a transaction."""
        try:
            with self._sessions.begin() as session:
                session.add(Booking(**booking.model_dump()))
        except IntegrityError as exc:
            if self.slot_taken(booking.booking_date, booking.booking_time):
                raise SlotUnavailableError("That date and time are already booked.") from exc
            raise DependencyError("The booking could not be stored.") from exc
        except SQLAlchemyError as exc:
            raise DependencyError("Booking storage is unavailable.") from exc

    def slot_taken(self, booking_date: date, booking_time: time) -> bool:
        """Read slot occupancy without claiming a reservation."""
        try:
            with self._sessions() as session:
                return (
                    session.scalar(
                        select(Booking.id)
                        .where(
                            Booking.booking_date == booking_date,
                            Booking.booking_time == booking_time,
                        )
                        .limit(1)
                    )
                    is not None
                )
        except SQLAlchemyError as exc:
            raise DependencyError("Booking storage is unavailable.") from exc
