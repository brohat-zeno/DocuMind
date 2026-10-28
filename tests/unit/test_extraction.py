"""File format and extraction tests."""

from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.core.exceptions import ExtractionError, UnsupportedFileTypeError, UploadTooLargeError
from app.services.text_extraction import extract_text


def pdf_bytes(text: str | None = None) -> bytes:
    """Build a minimal PDF fixture, optionally with extractable text."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_txt_encodings() -> None:
    """Accept UTF-8 BOM and a Latin-1 fallback."""
    assert extract_text("a.txt", "text/plain", b"\xef\xbb\xbfhello", 100) == "hello"
    assert extract_text("a.TXT", "text/plain; charset=latin1", b"caf\xe9", 100) == "café"


def test_pdf() -> None:
    """Extract actual PDF text."""
    assert "Hello" in extract_text("a.pdf", "application/pdf", pdf_bytes("Hello"), 10000)


@pytest.mark.parametrize("content", [b"", b"  \n"])
def test_empty_text(content: bytes) -> None:
    """Reject empty and whitespace-only text."""
    with pytest.raises(ExtractionError):
        extract_text("a.txt", "text/plain", content, 100)


@pytest.mark.parametrize("content", [b"not pdf", pdf_bytes()])
def test_unreadable_pdf(content: bytes) -> None:
    """Reject malformed and image-only PDFs."""
    with pytest.raises(ExtractionError):
        extract_text("a.pdf", "application/pdf", content, 10000)


def test_format_and_limit() -> None:
    """Reject mismatched MIME types, unsupported extensions, and oversized files."""
    for filename, mime in [("a.exe", "text/plain"), ("a.txt", "application/pdf")]:
        with pytest.raises(UnsupportedFileTypeError):
            extract_text(filename, mime, b"text", 100)
    with pytest.raises(UploadTooLargeError):
        extract_text("a.txt", "text/plain", b"abcd", 3)
