"""Dispatch by sniffed type. Every parser returns blocks, or raises RejectedFile."""

from __future__ import annotations

from collections.abc import Callable

from reconrag.config import Settings
from reconrag.domain import ParsedDocument
from reconrag.ingestion.parsers.email import parse_eml
from reconrag.ingestion.parsers.html import parse_html
from reconrag.ingestion.parsers.office import parse_docx, parse_xlsx
from reconrag.ingestion.parsers.pdf import parse_image, parse_pdf
from reconrag.ingestion.parsers.plain import parse_csv, parse_markdown, parse_txt

Parser = Callable[[bytes, Settings], ParsedDocument]

PARSERS: dict[str, Parser] = {
    "pdf": parse_pdf,
    "image": parse_image,
    "docx": parse_docx,
    "xlsx": parse_xlsx,
    "html": parse_html,
    "eml": parse_eml,
    "csv": parse_csv,
    "md": parse_markdown,
    "txt": parse_txt,
}
