"""Declarative ORM base."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Share metadata across persistent models."""
