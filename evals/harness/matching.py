"""Decides whether a chunk holds a piece of evidence.

Chunk ids differ per pipeline, so relevance is judged on text. Prose is matched fuzzily, because
parsers break lines and hyphenate. Numbers must match exactly, because sibling plans have the same
sentence with a different amount, and that difference is the whole point of the question.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from rapidfuzz import fuzz

from evals.harness.schema import Evidence

MATCH_THRESHOLD = 90.0
MIN_COVERAGE = 0.85

_NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)
_NUMBER = re.compile(r"\d+")
_LINE_BREAK_HYPHEN = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _LINE_BREAK_HYPHEN.sub("", text)
    return _NON_ALNUM.sub(" ", text.casefold()).replace("_", " ").strip()


def contains_evidence(chunk: str, quote: str, threshold: float = MATCH_THRESHOLD) -> bool:
    chunk_norm = normalize(chunk)
    quote_norm = normalize(quote)
    if not quote_norm or not chunk_norm:
        return False
    alignment = fuzz.partial_ratio_alignment(quote_norm, chunk_norm)
    if alignment is None or alignment.score < threshold:
        return False
    passage = chunk_norm[alignment.dest_start : alignment.dest_end]
    # A chunk shorter than the quote aligns the other way round, and would score 100 on a fragment.
    if len(passage) < MIN_COVERAGE * len(quote_norm):
        return False
    # Compare numbers only inside the matched passage. A chunk often has the same number elsewhere.
    return _NUMBER.findall(quote_norm) == _NUMBER.findall(passage)


def contains_table_row(chunk: str, row: str, value: str) -> bool:
    """Row label and value in one chunk. Parsers often put each cell on its own line."""
    if normalize(row) not in normalize(chunk):
        return False
    digits = re.sub(r"(?<=\d),(?=\d)", "", chunk)
    wanted = re.escape(value.replace(",", ""))
    return re.search(rf"(?<![\d.]){wanted}(?![\d])", digits) is not None


def evidence_matches(chunk: str, evidence: Evidence) -> bool:
    if evidence.quote:
        return contains_evidence(chunk, evidence.quote)
    return contains_table_row(chunk, evidence.row or "", evidence.value or "")


def evidence_rank(ranked_chunks: Sequence[tuple[str, str]], evidence: Evidence) -> int | None:
    """1-based rank of the first (file, text) pair from an allowed file that holds the evidence."""
    allowed = set(evidence.files)
    for rank, (filename, text) in enumerate(ranked_chunks, start=1):
        if filename in allowed and evidence_matches(text, evidence):
            return rank
    return None
