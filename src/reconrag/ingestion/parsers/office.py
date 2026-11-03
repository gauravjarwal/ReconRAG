"""Word and Excel. Both are zip files, so both pass the bomb check before opening."""

from __future__ import annotations

import datetime as dt
import io
import re
import zipfile
from typing import Any

import docx
from docx.document import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from reconrag.config import Settings
from reconrag.domain import Block, BlockKind, ParsedDocument
from reconrag.errors import RejectedFile
from reconrag.ingestion.sniff import check_zip

_HEADING_STYLE = re.compile(r"^(?:heading|überschrift|titre)\s*(\d)", re.I)


def _paragraph_block(paragraph: Paragraph) -> Block | None:
    text = paragraph.text.strip()
    if not text:
        return None
    style = paragraph.style.name if paragraph.style is not None and paragraph.style.name else ""
    if style.lower() == "title":
        return Block(BlockKind.HEADING, text, level=1)
    match = _HEADING_STYLE.match(style)
    if match:
        return Block(BlockKind.HEADING, text, level=int(match.group(1)))
    return Block(BlockKind.TEXT, text)


def _table_block(table: Table) -> Block | None:
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        previous = None
        for cell in row.cells:
            # Merged cells repeat the same underlying element; keep each one once.
            if previous is not None and cell._tc is previous:
                continue
            previous = cell._tc
            cells.append(" ".join(cell.text.split()))
        if any(cells):
            rows.append(cells)
    return Block(BlockKind.TABLE, rows=rows) if rows else None


def parse_docx(data: bytes, settings: Settings) -> ParsedDocument:
    check_zip(data, settings)
    try:
        document: Document = docx.Document(io.BytesIO(data))
        items = list(document.iter_inner_content())
        title = document.core_properties.title or ""
    except Exception as exc:  # python-docx raises zip, KeyError or lxml errors on bad input.
        raise RejectedFile("corrupt_file", "The Word file is corrupt.") from exc
    blocks: list[Block] = []
    for item in items:
        block = _paragraph_block(item) if isinstance(item, Paragraph) else _table_block(item)
        if block is not None:
            blocks.append(block)
    return ParsedDocument(blocks, parser="python-docx", title=title.strip())


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time(0) else value.isoformat(" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return " ".join(str(value).split())


def _split_regions(rows: list[list[str]]) -> list[list[list[str]]]:
    """Blank rows separate tables that share a sheet."""
    regions: list[list[list[str]]] = [[]]
    for row in rows:
        if any(row):
            regions[-1].append(row)
        elif regions[-1]:
            regions.append([])
    return [region for region in regions if region]


def _region_blocks(region: list[list[str]]) -> list[Block]:
    """Rows above the first one filling over half the columns are titles, not data."""
    width = max(sum(bool(cell) for cell in row) for row in region)
    start = next(i for i, row in enumerate(region) if sum(map(bool, row)) * 2 > width)
    blocks = [
        Block(BlockKind.TEXT, " ".join(cell for cell in row if cell)) for row in region[:start]
    ]
    table = region[start:]
    used = [i for i in range(len(table[0])) if any(row[i] for row in table)]
    trimmed = [[row[i] for i in used] for row in table]
    if len(trimmed) == 1 and sum(map(bool, trimmed[0])) == 1:
        blocks.append(Block(BlockKind.TEXT, " ".join(trimmed[0])))
    else:
        blocks.append(Block(BlockKind.TABLE, rows=trimmed))
    return blocks


def parse_xlsx(data: bytes, settings: Settings) -> ParsedDocument:
    check_zip(data, settings)
    try:
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except (InvalidFileException, zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        raise RejectedFile("corrupt_file", "The Excel file is corrupt.") from exc
    blocks: list[Block] = []
    warnings: list[str] = []
    try:
        for sheet in workbook.worksheets:
            if sheet.sheet_state != "visible":
                warnings.append("hidden_sheet_skipped")
                continue
            rows: list[list[str]] = []
            for values in sheet.iter_rows(values_only=True):
                rows.append([_cell(value) for value in values])
                if len(rows) >= settings.max_spreadsheet_rows:
                    warnings.append("rows_truncated")
                    break
            regions = _split_regions(rows)
            if regions:
                blocks.append(Block(BlockKind.HEADING, sheet.title, level=2))
                for region in regions:
                    blocks.extend(_region_blocks(region))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise RejectedFile("corrupt_file", "The Excel file is corrupt.") from exc
    finally:
        workbook.close()
    return ParsedDocument(blocks, parser="openpyxl", warnings=warnings)
