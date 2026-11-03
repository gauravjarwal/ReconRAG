"""Structure-aware chunking.

Sections are never mixed, tables stay whole with their headers, and every chunk knows its section
path and pages. Size is counted in tokens, using a regex close to what the embedding tokenizer
produces, because exact tokenizers would tie this module to one model.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from reconrag.config import Settings
from reconrag.domain import Block, BlockKind, Chunk, ParsedDocument
from reconrag.ingestion.text import normalize_for_hash
from reconrag.security.injection import looks_like_injection

_TOKEN = re.compile(r"\w+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_ABBREVIATION = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|no|fig|sec|inc|ltd|rs)\.$", re.IGNORECASE
)
_NUMERIC_CELL = re.compile(r"^[\s\d.,%$€£₹()+-]*$")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def text_hash(text: str) -> bytes:
    return hashlib.blake2b(normalize_for_hash(text).encode(), digest_size=16).digest()


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        piece = text[start : match.end()].strip()
        if piece and not _ABBREVIATION.search(piece):
            sentences.append(piece)
            start = match.end()
    tail = text[start:].strip()
    return [*sentences, tail] if tail else sentences


def _hard_split(text: str, limit: int) -> list[str]:
    """Last resort for a single sentence longer than a chunk."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        cost = count_tokens(word)
        if current and size + cost > limit:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += cost
    if current:
        pieces.append(" ".join(current))
    return pieces


def _units(block: Block, limit: int) -> list[str]:
    units: list[str] = []
    for line in block.text.split("\n"):
        for sentence in split_sentences(line):
            if count_tokens(sentence) > limit:
                units.extend(_hard_split(sentence, limit))
            else:
                units.append(sentence)
    return units


def _has_header(rows: list[list[str]]) -> bool:
    first = rows[0]
    if len(rows) < 2 or not all(first):
        return False
    return not any(_NUMERIC_CELL.match(cell) for cell in first)


def _render_row(header: list[str] | None, row: list[str]) -> str:
    if header is None:
        return " | ".join(cell for cell in row if cell)
    pairs = zip(header, row, strict=False)
    return "; ".join(f"{name}: {cell}" for name, cell in pairs if cell)


def table_chunks(rows: list[list[str]], limit: int) -> list[str]:
    """Rows repeat their column names, so a value is never separated from what it measures."""
    header = rows[0] if _has_header(rows) else None
    body = rows[1:] if header else rows
    if header and not body:
        body = [header]
        header = None
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for row in body:
        line = _render_row(header, row)
        cost = count_tokens(line)
        if current and size + cost > limit:
            chunks.append("\n".join(current))
            current, size = [], 0
        if cost > limit:
            chunks.extend(_hard_split(line, limit))
            continue
        current.append(line)
        size += cost
    if current:
        chunks.append("\n".join(current))
    return chunks


@dataclass
class _Buffer:
    """Sentences waiting to become a chunk, each with the page it came from."""

    units: list[tuple[str, int | None]] = field(default_factory=list)
    tokens: int = 0

    def add(self, text: str, page: int | None) -> None:
        self.units.append((text, page))
        self.tokens += count_tokens(text)

    def tail(self, limit: int) -> list[tuple[str, int | None]]:
        kept: list[tuple[str, int | None]] = []
        size = 0
        for unit in reversed(self.units):
            size += count_tokens(unit[0])
            if size > limit:
                break
            kept.insert(0, unit)
        return kept


class _Chunker:
    def __init__(self, title: str, settings: Settings) -> None:
        self.title = title
        self.settings = settings
        self.path: list[tuple[int, str]] = []
        self.chunks: list[Chunk] = []
        self.buffer = _Buffer()
        self.buffer_path: list[str] = []

    @property
    def section(self) -> list[str]:
        return [text for _, text in self.path]

    def _context(self, section: list[str]) -> str:
        parts = [self.title, *section] if self.title else section
        return " > ".join(dict.fromkeys(part for part in parts if part))

    def _emit(self, text: str, section: list[str], pages: list[int], kind: str) -> None:
        self.chunks.append(
            Chunk(
                ordinal=len(self.chunks),
                text=text,
                context=self._context(section),
                section_path=section,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                kind=kind,
                token_count=count_tokens(text),
                text_hash=text_hash(text),
                injection_risk=looks_like_injection(text),
            )
        )

    def flush(self, *, overlap: int = 0) -> None:
        if not self.buffer.units:
            return
        pages = [page for _, page in self.buffer.units if page is not None]
        text = " ".join(unit for unit, _ in self.buffer.units)
        self._emit(text, self.buffer_path, pages, "text")
        carried = self.buffer.tail(overlap) if overlap else []
        self.buffer = _Buffer()
        for unit, page in carried:
            self.buffer.add(unit, page)

    def add_heading(self, block: Block) -> None:
        # A stub section joins the next one, so it isn't left as an unusable tiny chunk.
        if self.buffer.tokens >= self.settings.chunk_min_tokens:
            self.flush()
        level = block.level or 1
        while self.path and self.path[-1][0] >= level:
            self.path.pop()
        self.path.append((level, block.text))
        if self.buffer.units:
            self.buffer.add(block.text + ":", block.page)
        else:
            self.buffer_path = self.section

    def add_text(self, block: Block) -> None:
        if not self.buffer.units:
            self.buffer_path = self.section
        for unit in _units(block, self.settings.chunk_target_tokens):
            size = count_tokens(unit)
            if self.buffer.tokens + size > self.settings.chunk_target_tokens:
                self.flush(overlap=self.settings.chunk_overlap_tokens)
            self.buffer.add(unit, block.page)

    def add_table(self, block: Block) -> None:
        self.flush()
        assert block.rows is not None
        pages = [block.page] if block.page is not None else []
        for text in table_chunks(block.rows, self.settings.table_max_tokens):
            self._emit(text, self.section, pages, "table")


def chunk_document(document: ParsedDocument, title: str, settings: Settings) -> list[Chunk]:
    chunker = _Chunker(title, settings)
    for block in document.blocks:
        if block.kind == BlockKind.HEADING:
            chunker.add_heading(block)
        elif block.kind == BlockKind.TABLE:
            chunker.add_table(block)
        else:
            chunker.add_text(block)
    chunker.flush()
    if not chunker.chunks and chunker.path:
        # A file with only headings still has text worth finding.
        chunker.buffer.add(" ".join(chunker.section), None)
        chunker.flush()
    return chunker.chunks
