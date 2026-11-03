"""HTML. Page furniture and hidden text are removed: hidden text is a common injection channel."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from reconrag.config import Settings
from reconrag.domain import Block, BlockKind, ParsedDocument
from reconrag.ingestion.text import decode

_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "object",
    "nav",
    "footer",
    "aside",
    "form",
    "button",
)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0", re.I)
_FURNITURE = re.compile(
    r"(^|[-_\s])(cookie|consent|newsletter|breadcrumbs?|sidebar|advert|promo|share|social)"
    r"([-_\s]|$)",
    re.I,
)
_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_LEAF_BLOCKS = {"p", "li", "pre", "blockquote", "dd", "dt", "figcaption", "summary", "address"}


def _is_noise(tag: Tag) -> bool:
    if tag.attrs is None:
        return False
    if tag.has_attr("hidden") or tag.get("aria-hidden") == "true":
        return True
    if _HIDDEN_STYLE.search(str(tag.get("style", ""))):
        return True
    names = f"{tag.get_attribute_list('class')} {tag.get('id', '')}"
    return bool(_FURNITURE.search(names))


def _table_rows(table: Tag) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells: list[str] = []
        for cell in row.find_all(["th", "td"], recursive=False):
            colspan = str(cell.get("colspan", "1"))
            span = int(colspan) if colspan.isdigit() else 1
            cells.extend([cell.get_text(" ", strip=True)] * min(span, 50))
        if any(cells):
            rows.append(cells)
    return rows


def _walk(node: Tag, blocks: list[Block]) -> None:
    inline: list[str] = []

    def flush() -> None:
        text = " ".join(" ".join(inline).split())
        if text:
            blocks.append(Block(BlockKind.TEXT, text))
        inline.clear()

    for child in node.children:
        if isinstance(child, NavigableString):
            inline.append(str(child))
        elif isinstance(child, Tag):
            if child.name in _HEADINGS:
                flush()
                text = child.get_text(" ", strip=True)
                if text:
                    blocks.append(Block(BlockKind.HEADING, text, level=_HEADINGS[child.name]))
            elif child.name == "table":
                flush()
                rows = _table_rows(child)
                if rows:
                    blocks.append(Block(BlockKind.TABLE, rows=rows))
            elif child.name in _LEAF_BLOCKS and not child.find(["table", "ul", "ol"]):
                flush()
                text = child.get_text(" ", strip=True)
                if text:
                    blocks.append(Block(BlockKind.TEXT, text))
            elif child.name in ("a", "b", "i", "em", "strong", "span", "code", "br", "sub", "sup"):
                inline.append(child.get_text(" ", strip=True))
            else:
                flush()
                _walk(child, blocks)
    flush()


def html_blocks(markup: str) -> tuple[list[Block], str]:
    soup = BeautifulSoup(markup, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()
    for tag in [t for t in soup.find_all(True) if _is_noise(t)]:
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[Block] = []
    _walk(root, blocks)  # type: ignore[arg-type]
    return blocks, title


def parse_html(data: bytes, settings: Settings) -> ParsedDocument:
    blocks, title = html_blocks(decode(data))
    return ParsedDocument(blocks, parser="html", title=title)
