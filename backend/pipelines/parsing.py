from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)


def parse_pdf(data: bytes) -> str:
    import PyPDF2

    reader = PyPDF2.PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_docx(data: bytes) -> str:
    import docx

    doc = docx.Document(io.BytesIO(data))
    return "\n".join(para.text for para in doc.paragraphs)


def parse_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


from typing import Callable

PARSERS: dict[str, Callable[[bytes], str]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_text,
    ".md": parse_text,
}

ALLOWED_EXTENSIONS = set(PARSERS.keys())

def parse_file(filename: str, data: bytes) -> str:
    """Extract raw text from a file given its name and bytes."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext!r}")
    text = parser(data)
    logger.debug("Parsed %s — %d chars", filename, len(text))
    return text
