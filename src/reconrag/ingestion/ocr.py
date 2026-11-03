"""OCR through RapidOCR. The model loads once per process, on first use."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

# Lines in one row overlap vertically by at least this share of their height.
_SAME_ROW = 0.5
# A vertical gap above this many line heights starts a new paragraph.
_PARAGRAPH_GAP = 1.2


@dataclass(frozen=True)
class OcrResult:
    lines: list[str]
    confidence: float


@lru_cache(maxsize=1)
def _engine() -> Any:
    from rapidocr import RapidOCR

    return RapidOCR(params={"Global.log_level": "error"})


def _rows(items: list[tuple[float, float, float, float, str]]) -> list[list[tuple]]:
    """Groups boxes into visual rows, so a table row stays together even with no table parser."""
    rows: list[list[tuple]] = []
    for item in sorted(items, key=lambda i: i[1]):
        top, bottom = item[1], item[3]
        if rows:
            last_top = min(i[1] for i in rows[-1])
            last_bottom = max(i[3] for i in rows[-1])
            overlap = min(bottom, last_bottom) - max(top, last_top)
            if overlap >= _SAME_ROW * min(bottom - top, last_bottom - last_top):
                rows[-1].append(item)
                continue
        rows.append([item])
    return rows


def _layout(boxes: Any, texts: tuple[str, ...]) -> list[str]:
    items = [
        (
            float(box[:, 0].min()),
            float(box[:, 1].min()),
            float(box[:, 0].max()),
            float(box[:, 1].max()),
            text,
        )
        for box, text in zip(boxes, texts, strict=True)
    ]
    lines: list[str] = []
    previous_bottom: float | None = None
    for row in _rows(items):
        top = min(i[1] for i in row)
        height = max(i[3] for i in row) - top
        if previous_bottom is not None and top - previous_bottom > _PARAGRAPH_GAP * height:
            lines.append("")
        lines.append("  ".join(i[4] for i in sorted(row, key=lambda i: i[0])))
        previous_bottom = max(i[3] for i in row)
    return lines


def ocr_image(image: np.ndarray) -> OcrResult:
    result = _engine()(image)
    if result is None or result.txts is None or len(result.txts) == 0:
        return OcrResult([], 0.0)
    confidence = float(np.mean(result.scores))
    return OcrResult(_layout(result.boxes, result.txts), confidence)
