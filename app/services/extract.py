"""Text extraction from uploaded files, page by page (pages are used for citations)."""

import io
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class ExtractionError(Exception):
    """The file is supported but its text could not be read."""


class UnsupportedFileError(Exception):
    pass


def extract_pages(filename: str, data: bytes) -> list[str]:
    """Return the text of each page. Text files count as a single page."""
    suffix = PurePath(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileError(f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    pages = _extract_pdf(data) if suffix == ".pdf" else [_decode_text(data)]

    if not any(page.strip() for page in pages):
        raise ExtractionError(
            "No text found in the file. Scanned PDFs need OCR, which is not supported."
        )
    return pages


def _extract_pdf(data: bytes) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("Encrypted PDFs are not supported")
        return [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise ExtractionError("The file is not a valid PDF") from exc


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")
