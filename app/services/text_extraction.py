"""File validation and text extraction."""

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import ExtractionError, UnsupportedFileTypeError, UploadTooLargeError


def extract_text(filename: str, content_type: str, content: bytes, max_bytes: int) -> str:
    """Validate an upload and return its extracted text."""
    extension = Path(filename).suffix.lower()
    expected_type = {".pdf": "application/pdf", ".txt": "text/plain"}.get(extension)
    if expected_type is None or content_type.split(";")[0].strip().lower() != expected_type:
        raise UnsupportedFileTypeError("Use a .pdf (application/pdf) or .txt (text/plain) file.")
    if len(content) > max_bytes:
        raise UploadTooLargeError(f"File exceeds the {max_bytes}-byte limit.")
    if not content:
        raise ExtractionError("The uploaded file is empty.")
    if extension == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
    else:
        try:
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ExtractionError(
                "The PDF could not be read; use an unencrypted, valid PDF."
            ) from exc
    if not text.strip():
        raise ExtractionError("The document has no extractable text.")
    return text
