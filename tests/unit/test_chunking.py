"""Chunk size, boundary, and lossless overlap tests."""

import pytest

from app.core.exceptions import InvalidChunkingError
from app.services.chunking import Chunker, FixedSizeChunker, RecursiveChunker


@pytest.mark.parametrize("chunker", [FixedSizeChunker(), RecursiveChunker()])
@pytest.mark.parametrize("text", ["", "a", "abc", "x" * 101, "One. Two!\n\nThree four five.\n" * 8])
@pytest.mark.parametrize("size,overlap", [(1, 0), (8, 0), (8, 3), (8, 7), (40, 5)])
def test_lossless_chunks(chunker: Chunker, text: str, size: int, overlap: int) -> None:
    """Keep all characters, bounded length, and exact overlap."""
    chunks = chunker.chunk(text, size, overlap)
    assert all(0 < len(chunk) <= size for chunk in chunks)
    rebuilt = chunks[0] if chunks else ""
    for previous, current in zip(chunks, chunks[1:], strict=False):
        if overlap:
            assert previous[-overlap:] == current[:overlap]
        rebuilt += current[overlap:]
    assert rebuilt == text


@pytest.mark.parametrize("chunker", [FixedSizeChunker(), RecursiveChunker()])
@pytest.mark.parametrize("size,overlap", [(0, 0), (3, -1), (3, 3), (3, 4)])
def test_invalid_parameters(chunker: Chunker, size: int, overlap: int) -> None:
    """Reject invalid parameters before partitioning."""
    with pytest.raises(InvalidChunkingError):
        chunker.chunk("abc", size, overlap)


def test_fixed_windows() -> None:
    """Use exact strides without redundant tails."""
    assert FixedSizeChunker().chunk("abcdefghij", 6, 2) == ["abcdef", "efghij"]


def test_recursive_boundaries() -> None:
    """Prefer intact paragraph and sentence pieces when they fit."""
    assert RecursiveChunker().chunk("One. Two. Three.", 10, 0) == ["One. Two. ", "Three."]
    assert RecursiveChunker().chunk("Alpha\n\nBeta\n\nGamma", 12, 0) == [
        "Alpha\n\n",
        "Beta\n\nGamma",
    ]
