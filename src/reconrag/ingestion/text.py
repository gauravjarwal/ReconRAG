"""Repairs text: decoding, mojibake, invisible characters, and lines broken by layout."""

from __future__ import annotations

import re
import unicodedata

import ftfy
from charset_normalizer import from_bytes

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_SPACES = re.compile(r"[ \t  -   　]+")
_BROKEN_WORD = re.compile(r"(?<=[^\W\d_])-\s*\n\s*(?=[a-z])")
_BLANK_RUN = re.compile(r"\n{3,}")


def decode(data: bytes) -> str:
    """UTF-8 first because it almost never decodes by accident; then a detected encoding."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    best = from_bytes(data[:200_000]).best()
    return data.decode(best.encoding if best else "cp1252", errors="replace")


def _strip_control(text: str) -> str:
    text = text.translate(_ZERO_WIDTH)
    return "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")


def clean_text(text: str) -> str:
    text = ftfy.fix_text(text, unescape_html=False)
    text = unicodedata.normalize(
        "NFC", _strip_control(text.replace("\r\n", "\n").replace("\r", "\n"))
    )
    text = _BROKEN_WORD.sub("", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def normalize_for_hash(text: str) -> str:
    """Two documents with the same words should hash the same, whatever their layout."""
    return " ".join(text.casefold().split())
