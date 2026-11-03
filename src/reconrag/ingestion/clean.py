"""Cleans parsed documents. Works on blocks, so structure survives."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from reconrag.domain import Block, BlockKind, ParsedDocument
from reconrag.ingestion.text import clean_text

_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+(?:\s*(?:/|of)\s*\d+)?$", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_MARGIN_BLOCKS = 2
_MIN_PAGES = 3
# A running header is short. Long lines that repeat are content, such as form-style pages.
_MAX_MARGIN_WORDS = 10


def _signature(text: str) -> str:
    """Page numbers change on every page, so digits don't count when spotting repeats."""
    return _DIGITS.sub("#", " ".join(text.casefold().split()))


def _clean_block(block: Block) -> Block | None:
    if block.rows is not None:
        rows = [[clean_text(cell) for cell in row] for row in block.rows]
        rows = [row for row in rows if any(row)]
        if not rows:
            return None
        block.rows = rows
        return block
    block.text = clean_text(block.text)
    return block if block.text else None


def _repeated_margins(blocks: list[Block]) -> set[str]:
    """Text in the first or last blocks of most pages is a running header or footer."""
    by_page: dict[int, list[Block]] = defaultdict(list)
    for block in blocks:
        if block.page is not None and block.rows is None:
            by_page[block.page].append(block)
    if len(by_page) < _MIN_PAGES:
        return set()
    seen: Counter[str] = Counter()
    for page_blocks in by_page.values():
        margins = page_blocks[:_MARGIN_BLOCKS] + page_blocks[-_MARGIN_BLOCKS:]
        seen.update(
            _signature(block.text)
            for block in {id(b): b for b in margins}.values()
            if len(block.text.split()) <= _MAX_MARGIN_WORDS
        )
    threshold = max(_MIN_PAGES, len(by_page) // 2 + 1)
    return {text for text, count in seen.items() if count >= threshold}


def clean_document(document: ParsedDocument) -> ParsedDocument:
    blocks = [cleaned for block in document.blocks if (cleaned := _clean_block(block))]
    repeated = _repeated_margins(blocks)
    kept: list[Block] = []
    for block in blocks:
        if block.rows is None:
            if _signature(block.text) in repeated or _PAGE_NUMBER.match(block.text):
                continue
        kept.append(block)
    document.blocks = kept
    document.title = clean_text(document.title)
    return document


def has_content(document: ParsedDocument) -> bool:
    return any(block.kind != BlockKind.HEADING or block.text for block in document.blocks)
