"""Hand-written character and boundary-aware text chunking."""

import re
from typing import Protocol

from app.core.exceptions import InvalidChunkingError


class Chunker(Protocol):
    """Partition text into bounded overlapping chunks."""

    def chunk(self, text: str, size: int, overlap: int) -> list[str]:
        """Return chunks without dropping document characters."""
        ...


def validate_chunking(size: int, overlap: int) -> None:
    """Require a positive size and an overlap smaller than it."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise InvalidChunkingError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")


class FixedSizeChunker:
    """Split text using fixed character windows."""

    def chunk(self, text: str, size: int, overlap: int) -> list[str]:
        """Return fixed windows, avoiding a redundant trailing chunk."""
        validate_chunking(size, overlap)
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
        return chunks


class RecursiveChunker:
    """Prefer paragraph, sentence, then word boundaries."""

    def _pieces(self, text: str, size: int, level: int = 0) -> list[str]:
        separators = (r"(?<=\n)\n+", r"(?<=[.!?])\s+", r"\s+")
        if len(text) <= size:
            return [text] if text else []
        if level == len(separators):
            return [text[start : start + size] for start in range(0, len(text), size)]
        pieces: list[str] = []
        start = 0
        for match in re.finditer(separators[level], text):
            end = match.end()
            pieces.extend(self._pieces(text[start:end], size, level + 1))
            start = end
        pieces.extend(self._pieces(text[start:], size, level + 1))
        return pieces

    def chunk(self, text: str, size: int, overlap: int) -> list[str]:
        """Merge recursive pieces with exact character overlap."""
        validate_chunking(size, overlap)
        pieces = self._pieces(text, size)
        boundaries: list[int] = []
        position = 0
        for piece in pieces:
            position += len(piece)
            boundaries.append(position)
        chunks: list[str] = []
        start = 0
        boundary_index = 0
        while start < len(text):
            limit = min(start + size, len(text))
            while (
                boundary_index < len(boundaries) and boundaries[boundary_index] <= start + overlap
            ):
                boundary_index += 1
            end = limit
            index = boundary_index
            while index < len(boundaries) and boundaries[index] <= limit:
                end = boundaries[index]
                index += 1
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
        return chunks
