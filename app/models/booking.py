"""Confirmed interview bookings."""

from datetime import date, datetime, time
from uuid import UUID

from sqlalchemy import Date, DateTime, String, Time, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Booking(Base):
    """Reserve a date and time uniquely across all sessions."""

    __tablename__ = "bookings"
    __table_args__ = (UniqueConstraint("booking_date", "booking_time", name="uq_booking_slot"),)
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String)
    booking_date: Mapped[date] = mapped_column(Date)
    booking_time: Mapped[time] = mapped_column(Time)
    session_id: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
