"""Duplicate and version detection.

Identical bytes or identical text are dropped. Near-duplicates are kept and linked, because two
near-identical documents can differ in the one number that matters.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePath

from reconrag.ingestion.text import normalize_for_hash

_SHINGLE = 3
_BITS = 64
_WORD = re.compile(r"\w+")
_VERSION_WORDS = re.compile(
    r"(?:^|[\s_.\-])(?:v(?:er(?:sion)?)?[\s_.\-]?\d+(?:\.\d+)*|rev(?:ision)?[\s_.\-]?\d*|"
    r"final|draft|latest|updated|copy|new|old)(?=$|[\s_.\-])",
    re.IGNORECASE,
)
_COPY_SUFFIX = re.compile(r"\s*\(\d+\)$")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?:[-_.]?(0[1-9]|1[0-2]))?(?:[-_.]?([0-3]\d))?(?!\d)")
_SEPARATORS = re.compile(r"[\s_.\-]+")


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def text_digest(text: str) -> bytes:
    return hashlib.sha256(normalize_for_hash(text).encode()).digest()


def simhash(text: str) -> int:
    """64-bit fingerprint, signed to fit Postgres bigint. Similar text gives close fingerprints."""
    words = _WORD.findall(text.casefold())
    votes = [0] * _BITS
    for index in range(max(1, len(words) - _SHINGLE + 1)):
        shingle = " ".join(words[index : index + _SHINGLE])
        digest = hashlib.blake2b(shingle.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(_BITS):
            votes[bit] += 1 if value >> bit & 1 else -1
    fingerprint = sum(1 << bit for bit in range(_BITS) if votes[bit] > 0)
    return fingerprint - (1 << 64) if fingerprint >= 1 << 63 else fingerprint


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & ((1 << 64) - 1)).bit_count()


def document_key(filename: str) -> str:
    """Names that differ only by version words or copy markers are the same document."""
    stem = _COPY_SUFFIX.sub("", PurePath(filename).stem)
    previous = None
    while previous != stem:
        previous = stem
        stem = _VERSION_WORDS.sub("", stem)
    return _SEPARATORS.sub(" ", stem).strip().casefold() or PurePath(filename).stem.casefold()


def edition(filename: str) -> int:
    """A date in the name as YYYYMMDD, or 0. Months and days default to the start of the period."""
    matches = _YEAR.findall(PurePath(filename).stem)
    if not matches:
        return 0
    year, month, day = matches[-1]
    return int(year) * 10_000 + int(month or 1) * 100 + int(day or 1)


def undated_key(key: str) -> str:
    """The key without its date, for spotting a new edition of the same series."""
    return _SEPARATORS.sub(" ", _YEAR.sub("", key)).strip()
