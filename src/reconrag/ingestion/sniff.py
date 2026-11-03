"""Decides what a file really is from its bytes. The extension is only a hint."""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePath

from reconrag.config import Settings
from reconrag.errors import RejectedFile

_MAX_ZIP_ENTRIES = 20_000
# Small archives may compress well; the ratio check only matters once the output is large.
_RATIO_CHECK_ABOVE = 10 * 1024 * 1024

_BINARY_SIGNATURES = (
    (b"MZ", "executable"),
    (b"\x7fELF", "executable"),
    (b"\xfe\xed\xfa", "executable"),
    (b"\xcf\xfa\xed\xfe", "executable"),
    (b"\xd0\xcf\x11\xe0", "legacy Office format"),
    (b"Rar!", "archive"),
    (b"7z\xbc\xaf", "archive"),
    (b"\x1f\x8b", "archive"),
)
_IMAGE_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"II*\x00",
    b"MM\x00*",
)
_TEXT_EXTENSIONS = {
    ".txt": "txt",
    ".text": "txt",
    ".log": "txt",
    ".md": "md",
    ".markdown": "md",
    ".csv": "csv",
    ".tsv": "csv",
    ".html": "html",
    ".htm": "html",
    ".eml": "eml",
}
_BINARY_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def check_zip(data: bytes, settings: Settings) -> zipfile.ZipFile:
    """Reads the central directory only, so a bomb is refused before anything is inflated."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        entries = archive.infolist()
    except (zipfile.BadZipFile, ValueError) as exc:
        raise RejectedFile("corrupt_file", "The archive is corrupt.") from exc
    total = sum(entry.file_size for entry in entries)
    too_dense = total > _RATIO_CHECK_ABOVE and total > settings.max_zip_ratio * len(data)
    if len(entries) > _MAX_ZIP_ENTRIES or total > settings.max_unzipped_bytes or too_dense:
        raise RejectedFile("zip_bomb", "The file expands too far to be processed safely.")
    return archive


def _zip_kind(data: bytes, settings: Settings) -> str:
    names = set(check_zip(data, settings).namelist())
    if "word/document.xml" in names:
        return "docx"
    if "xl/workbook.xml" in names:
        return "xlsx"
    raise RejectedFile("unsupported_type", "This archive isn't a Word or Excel file.")


def _looks_binary(data: bytes) -> bool:
    sample = data[:8192]
    if sample.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf")):
        return False
    if b"\x00" in sample:
        return True
    control = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return control > len(sample) * 0.1


def _text_kind(sample: str, extension: str) -> str | None:
    lowered = sample.lstrip("﻿ \r\n\t").lower()
    if lowered.startswith(("<!doctype html", "<html")):
        return "html"
    return _TEXT_EXTENSIONS.get(extension)


def sniff(data: bytes, filename: str, settings: Settings) -> str:
    """Returns one of pdf, docx, xlsx, image, html, eml, csv, md, txt, or raises RejectedFile."""
    if not data.strip():
        raise RejectedFile("empty_file", "The file is empty.")
    extension = PurePath(filename).suffix.lower()
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if data.startswith(b"PK\x03\x04"):
        return _zip_kind(data, settings)
    if data.startswith(_IMAGE_SIGNATURES):
        return "image"
    for signature, label in _BINARY_SIGNATURES:
        if data.startswith(signature):
            raise RejectedFile("unsupported_type", f"Files of this kind aren't accepted: {label}.")
    if _looks_binary(data):
        raise RejectedFile("unsupported_type", "The file's content isn't a supported format.")
    if extension in _BINARY_EXTENSIONS:
        raise RejectedFile("type_mismatch", f"The content doesn't match the {extension} extension.")
    kind = _text_kind(data[:2048].decode("utf-8", errors="ignore"), extension)
    if kind is None:
        raise RejectedFile("unsupported_type", f"Files of type {extension!r} aren't accepted.")
    return kind
