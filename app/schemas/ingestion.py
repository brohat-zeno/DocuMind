"""Document and vector data contracts."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(StrEnum):
    """Supported text partition strategies."""

    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"


class DocumentCreated(BaseModel):
    """Return successful ingestion details."""

    document_id: UUID
    filename: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int


class DocumentMetadata(BaseModel):
    """Represent persisted document metadata."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    filename: str
    content_type: str
    size_bytes: int
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    created_at: datetime


class VectorPayload(BaseModel):
    """Describe one indexed document chunk."""

    document_id: UUID
    chunk_index: int
    text: str
    filename: str
    chunking_strategy: ChunkingStrategy


class VectorChunk(BaseModel):
    """Carry a chunk and its embedding to the vector repository."""

    id: UUID
    payload: VectorPayload
    embedding: list[float]


class RetrievedChunk(BaseModel):
    """Carry a scored search result."""

    payload: VectorPayload
    score: float = Field(ge=-1, le=1)
