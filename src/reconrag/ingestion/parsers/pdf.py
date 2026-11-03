"""PDF and images. Each page is routed on its own: text layer if usable, otherwise OCR.

PyMuPDF is AGPL. It is used only in this module, so it can be swapped. See ADR 0005.
"""

from __future__ import annotations

import io
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymupdf
from PIL import Image, ImageSequence, UnidentifiedImageError

from reconrag.config import Settings
from reconrag.domain import Block, BlockKind, ParsedDocument
from reconrag.errors import RejectedFile
from reconrag.ingestion.ocr import OcrResult, ocr_image
from reconrag.ingestion.quality import garble_ratio

_HEADING_SIZE_RATIO = 1.15
_MAX_HEADING_WORDS = 14
_FULL_WIDTH = 0.6
_MAX_GARBLE = 0.3
_BOLD_FLAG = 16
_MAX_IMAGE_PIXELS = 100_000_000


@dataclass
class _Item:
    """A positioned piece of a page: a text block or a table."""

    x0: float
    y0: float
    x1: float
    y1: float
    block: Block


@dataclass
class _Ocr:
    pages: int = 0
    confidences: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def reading_order(items: list[_Item], page_width: float) -> list[_Item]:
    """Top to bottom, with narrow blocks between full-width ones read as two columns."""
    middle = page_width / 2
    narrow = [i for i in items if i.x1 - i.x0 < _FULL_WIDTH * page_width]
    left = sum(i.x1 <= middle + 10 for i in narrow)
    right = sum(i.x0 >= middle - 10 for i in narrow)
    if left < 2 or right < 2:
        return sorted(items, key=lambda i: (round(i.y0), i.x0))

    ordered: list[_Item] = []
    band: list[_Item] = []

    def flush() -> None:
        ordered.extend(sorted(band, key=lambda i: (i.x0 >= middle - 10, i.y0)))
        band.clear()

    for item in sorted(items, key=lambda i: i.y0):
        if item.x1 - item.x0 >= _FULL_WIDTH * page_width:
            flush()
            ordered.append(item)
        else:
            band.append(item)
    flush()
    return ordered


def _body_size(document: pymupdf.Document) -> float:
    sizes: Counter[float] = Counter()
    for index in range(document.page_count):
        for block in document[index].get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _line_text(line: dict[str, Any]) -> str:
    return "".join(span["text"] for span in line["spans"]).strip()


def _text_block(block: dict[str, Any], body_size: float, page_number: int) -> Block | None:
    lines = [text for line in block.get("lines", []) if (text := _line_text(line))]
    if not lines:
        return None
    text = "\n".join(lines)
    spans = [span for line in block["lines"] for span in line["spans"] if span["text"].strip()]
    size = statistics.median(span["size"] for span in spans)
    bold = all(span["flags"] & _BOLD_FLAG for span in spans)
    short = len(text.split()) <= _MAX_HEADING_WORDS and not text.rstrip().endswith((".", ";"))
    if short and body_size and size >= body_size * _HEADING_SIZE_RATIO:
        ratio = size / body_size
        return Block(
            BlockKind.HEADING,
            " ".join(lines),
            page=page_number,
            level=1 if ratio > 1.6 else 2 if ratio > 1.3 else 3,
        )
    if short and bold:
        return Block(BlockKind.HEADING, " ".join(lines), page=page_number, level=3)
    return Block(BlockKind.TEXT, _unwrap(lines), page=page_number)


def _unwrap(lines: list[str]) -> str:
    """Layout line breaks aren't meaningful, but a word broken at a hyphen must stay joined."""
    text = lines[0]
    for line in lines[1:]:
        joiner = "\n" if text.endswith("-") else " "
        text += joiner + line
    return text


def _tables(page: pymupdf.Page, page_number: int) -> list[_Item]:
    try:
        found = page.find_tables().tables
    except Exception:  # find_tables fails on some malformed drawings; the text still extracts.
        return []
    items: list[_Item] = []
    for table in found:
        rows = [
            [" ".join((cell or "").split()) for cell in row] for row in table.extract() if any(row)
        ]
        if len(rows) >= 2 and len(rows[0]) >= 2:
            x0, y0, x1, y1 = table.bbox
            block = Block(BlockKind.TABLE, page=page_number, rows=rows)
            items.append(_Item(x0, y0, x1, y1, block))
    return items


def _inside(bbox: tuple[float, ...], tables: list[_Item]) -> bool:
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return any(t.x0 <= x <= t.x1 and t.y0 <= y <= t.y1 for t in tables)


def _text_page(page: pymupdf.Page, page_number: int, body_size: float) -> list[Block]:
    tables = _tables(page, page_number)
    items = list(tables)
    for raw in page.get_text("dict", sort=False)["blocks"]:
        if raw.get("type") != 0 or _inside(raw["bbox"], tables):
            continue
        block = _text_block(raw, body_size, page_number)
        if block is not None:
            x0, y0, x1, y1 = raw["bbox"]
            items.append(_Item(x0, y0, x1, y1, block))
    return [item.block for item in reading_order(items, page.rect.width)]


def _needs_ocr(page: pymupdf.Page, settings: Settings) -> bool:
    text = page.get_text("text")
    if len(text.strip()) < settings.min_page_chars:
        return bool(page.get_images(full=False))
    return garble_ratio(text) > _MAX_GARBLE


def _render(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8)
    return image.reshape(pixmap.height, pixmap.width, 3)


def _ocr_blocks(result: OcrResult, page_number: int | None) -> list[Block]:
    groups: list[list[str]] = [[]]
    for line in result.lines:
        if line:
            groups[-1].append(line)
        elif groups[-1]:
            groups.append([])
    return [Block(BlockKind.TEXT, "\n".join(group), page=page_number) for group in groups if group]


def _ocr_page(page: pymupdf.Page, page_number: int, settings: Settings, ocr: _Ocr) -> list[Block]:
    result = ocr_image(_render(page, settings.ocr_dpi))
    if not result.lines:
        ocr.warnings.append(f"ocr_empty_page:{page_number}")
        return []
    ocr.pages += 1
    ocr.confidences.append(result.confidence)
    return _ocr_blocks(result, page_number)


def parse_pdf(data: bytes, settings: Settings) -> ParsedDocument:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # MuPDF raises several unrelated error types on bad input.
        raise RejectedFile("corrupt_file", "The PDF is corrupt.") from exc
    with document:
        if document.needs_pass:
            raise RejectedFile("encrypted_file", "The PDF is password protected.")
        if document.page_count > settings.max_pages:
            raise RejectedFile("too_many_pages", f"The PDF has over {settings.max_pages} pages.")
        if document.page_count == 0:
            raise RejectedFile("empty_file", "The PDF has no pages.")
        title = (document.metadata or {}).get("title") or ""
        body_size = _body_size(document)
        blocks: list[Block] = []
        ocr = _Ocr()
        for index in range(document.page_count):
            page = document[index]
            number = index + 1
            if _needs_ocr(page, settings):
                if settings.ocr_enabled:
                    blocks.extend(_ocr_page(page, number, settings, ocr))
                else:
                    ocr.warnings.append(f"ocr_disabled_page:{number}")
            else:
                blocks.extend(_text_page(page, number, body_size))
        return ParsedDocument(
            blocks,
            parser="pymupdf",
            title=title.strip(),
            page_count=document.page_count,
            ocr_pages=ocr.pages,
            ocr_confidence=statistics.mean(ocr.confidences) if ocr.confidences else None,
            warnings=ocr.warnings,
        )


def parse_image(data: bytes, settings: Settings) -> ParsedDocument:
    if not settings.ocr_enabled:
        raise RejectedFile("ocr_disabled", "Images need OCR, which is turned off.")
    Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
    blocks: list[Block] = []
    confidences: list[float] = []
    try:
        image = Image.open(io.BytesIO(data))
        frames = list(ImageSequence.Iterator(image))
        for number, frame in enumerate(frames[: settings.max_pages], start=1):
            result = ocr_image(np.asarray(frame.convert("RGB")))
            if result.lines:
                confidences.append(result.confidence)
                blocks.extend(_ocr_blocks(result, number))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RejectedFile("corrupt_file", "The image can't be read.") from exc
    return ParsedDocument(
        blocks,
        parser="rapidocr",
        page_count=len(frames),
        ocr_pages=len(confidences),
        ocr_confidence=statistics.mean(confidences) if confidences else None,
    )
