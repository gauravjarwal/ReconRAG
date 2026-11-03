"""TXT, Markdown and CSV."""

from __future__ import annotations

import csv
import io
import re

from reconrag.config import Settings
from reconrag.domain import Block, BlockKind, ParsedDocument
from reconrag.ingestion.text import decode

_ATX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_MD_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_NUMBERED_HEADING = re.compile(r"^(?:section\s+)?\d+(?:\.\d+)*[.)]?\s+[A-Z][^.!?]*$", re.IGNORECASE)
_MAX_HEADING_WORDS = 12


def _paragraphs(text: str) -> list[list[str]]:
    groups: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.strip():
            groups[-1].append(line.rstrip())
        elif groups[-1]:
            groups.append([])
    return [group for group in groups if group]


def _join_wrapped(lines: list[str]) -> str:
    """Hard-wrapped lines become one paragraph; list items stay on their own lines."""
    if any(_LIST_ITEM.match(line) for line in lines):
        return "\n".join(lines)
    return " ".join(line.strip() for line in lines)


def _is_plain_heading(lines: list[str]) -> bool:
    if len(lines) != 1 or len(lines[0].split()) > _MAX_HEADING_WORDS:
        return False
    line = lines[0].strip().rstrip(":")
    upper = line.isupper() and any(c.isalpha() for c in line)
    return upper or bool(_NUMBERED_HEADING.match(line))


def parse_txt(data: bytes, settings: Settings) -> ParsedDocument:
    blocks = [
        Block(BlockKind.HEADING, group[0].strip().rstrip(":"), level=2)
        if _is_plain_heading(group)
        else Block(BlockKind.TEXT, _join_wrapped(group))
        for group in _paragraphs(decode(data))
    ]
    return ParsedDocument(blocks, parser="txt")


def _md_table(lines: list[str]) -> list[list[str]] | None:
    if len(lines) < 2 or "|" not in lines[0] or not _MD_RULE.match(lines[1]):
        return None
    rows = [line.strip().strip("|").split("|") for line in [lines[0], *lines[2:]]]
    return [[cell.strip() for cell in row] for row in rows]


def _md_group(lines: list[str]) -> list[Block]:
    table = _md_table(lines)
    if table:
        return [Block(BlockKind.TABLE, rows=table)]
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(Block(BlockKind.TEXT, _join_wrapped(buffer)))
            buffer.clear()

    for index, line in enumerate(lines):
        atx = _ATX.match(line)
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if atx:
            flush()
            blocks.append(Block(BlockKind.HEADING, atx.group(2), level=len(atx.group(1))))
        elif _SETEXT.match(following) and line.strip():
            flush()
            level = 1 if following.startswith("=") else 2
            blocks.append(Block(BlockKind.HEADING, line.strip(), level=level))
        elif not _SETEXT.match(line):
            buffer.append(line)
    flush()
    return blocks


def parse_markdown(data: bytes, settings: Settings) -> ParsedDocument:
    text = _FRONT_MATTER.sub("", decode(data).replace("\r\n", "\n"))
    blocks = [block for group in _paragraphs(text) for block in _md_group(group)]
    title = next((b.text for b in blocks if b.kind == BlockKind.HEADING and b.level == 1), "")
    return ParsedDocument(blocks, parser="markdown", title=title)


def parse_csv(data: bytes, settings: Settings) -> ParsedDocument:
    text = decode(data)
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(
            text[:4096], delimiters=",;\t|"
        )
    except csv.Error:
        dialect = csv.excel
    csv.field_size_limit(1_000_000)
    rows: list[list[str]] = []
    warnings: list[str] = []
    try:
        for row in csv.reader(io.StringIO(text), dialect):
            if any(cell.strip() for cell in row):
                rows.append([cell.strip() for cell in row])
            if len(rows) >= settings.max_spreadsheet_rows:
                warnings.append("rows_truncated")
                break
    except csv.Error:
        warnings.append("csv_malformed")
    blocks = [Block(BlockKind.TABLE, rows=rows)] if rows else []
    return ParsedDocument(blocks, parser="csv", warnings=warnings)
