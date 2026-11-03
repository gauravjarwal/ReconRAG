"""Scores extracted text, so bad extractions are flagged rather than silently indexed."""

from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect

from reconrag.domain import ParsedDocument

DetectorFactory.seed = 0

_COMMON = set(" \n\t.,;:!?'\"()[]{}-–—/%&+=$€£₹@#*_|<>\\°")
_WORD = re.compile(r"[^\W\d_]{2,}")
_MIN_CHARS = 100


def garble_ratio(text: str) -> float:
    """Share of characters that aren't letters, digits or normal punctuation."""
    if not text:
        return 0.0
    odd = sum(not (c.isalnum() or c in _COMMON) for c in text)
    return odd / len(text)


def _word_shape(text: str) -> float:
    """OCR noise and broken encodings produce runs of one- and two-letter fragments."""
    tokens = text.split()
    if not tokens:
        return 0.0
    words = sum(
        bool(_WORD.search(token)) or token[:1].isdigit() or token.casefold() in ("a", "i")
        for token in tokens
    )
    return words / len(tokens)


def document_text(document: ParsedDocument) -> str:
    parts: list[str] = []
    for block in document.blocks:
        if block.rows is not None:
            parts.extend(" ".join(row) for row in block.rows)
        else:
            parts.append(block.text)
    return "\n".join(parts)


def score(document: ParsedDocument) -> tuple[float, list[str]]:
    text = document_text(document)
    flags: list[str] = []
    if len(text) < _MIN_CHARS:
        flags.append("short_text")
    value = (1.0 - min(1.0, garble_ratio(text) * 5)) * _word_shape(text)
    if document.ocr_confidence is not None:
        value *= document.ocr_confidence
        flags.append("ocr")
    if document.warnings:
        flags.extend(sorted({warning.split(":")[0] for warning in document.warnings}))
    return round(value, 4), flags


def language(text: str) -> str:
    try:
        return detect(text[:2000]) if len(text.strip()) >= 20 else ""
    except LangDetectException:
        return ""
