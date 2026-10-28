"""SQL document metadata."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Document(Base):
    """Persist metadata for an indexed document."""

    __tablename__ = "documents"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    filename: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(String)
    size_bytes: Mapped[int]
    chunking_strategy: Mapped[str] = mapped_column(String)
    chunk_size: Mapped[int]
    chunk_overlap: Mapped[int]
    chunk_count: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
