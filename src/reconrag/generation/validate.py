"""Checks a finished answer against its sources.

Numeric grounding is the strongest check here, and it only covers numbers: every number stated
must appear in a source the sentence cites. It says nothing about claims that hold no number.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from reconrag.security.phi_detect import detect_structured_phi

_CITATION = re.compile(r"\[(S\d+)\]")
_NUMBER = re.compile(r"(?<![\w.])[$€£₹]?\d[\d,]*(?:\.\d+)?%?(?![\w])")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z_]*_[0-9a-fA-F]{12}\]")


@dataclass
class Validated:
    answer: str
    cited: list[str]
    warnings: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)


def _canonical(number: str) -> str:
    value = number.strip("$€£₹%").replace(",", "")
    return value.rstrip("0").rstrip(".") if "." in value else value


def _numbers(text: str) -> set[str]:
    return {_canonical(match) for match in _NUMBER.findall(_PLACEHOLDER.sub(" ", text))}


def strip_bad_citations(answer: str, refs: Mapping[str, str]) -> tuple[str, int]:
    removed = 0

    def check(match: re.Match[str]) -> str:
        nonlocal removed
        if match.group(1) in refs:
            return match.group(0)
        removed += 1
        return ""

    return _CITATION.sub(check, answer), removed


def ungrounded_numbers(answer: str, refs: Mapping[str, str]) -> list[str]:
    """A cited sentence is checked against its citations; an uncited one against every source."""
    every_source = " ".join(refs.values())
    missing: list[str] = []
    for sentence in _SENTENCE_BREAK.split(answer):
        cited = _CITATION.findall(sentence)
        allowed = _numbers(" ".join(refs[ref] for ref in cited if ref in refs) or every_source)
        missing.extend(sorted(_numbers(_CITATION.sub(" ", sentence)) - allowed))
    return list(dict.fromkeys(missing))


def redact_unsourced_phi(answer: str, sources: str) -> tuple[str, int]:
    """An identifier that isn't in the sources was invented or recalled, so it doesn't go out."""
    redacted = 0
    for span in reversed(detect_structured_phi(answer)):
        if span.text not in sources:
            answer = f"{answer[: span.start]}[{span.entity_type} REDACTED]{answer[span.end :]}"
            redacted += 1
    return answer, redacted


def validate_answer(answer: str, refs: Mapping[str, str], max_chars: int) -> Validated:
    warnings: list[str] = []
    answer, removed = strip_bad_citations(answer, refs)
    if removed:
        warnings.append("citations_removed")
    answer, redacted = redact_unsourced_phi(answer, " ".join(refs.values()))
    if redacted:
        warnings.append("phi_redacted")
    unverified = ungrounded_numbers(answer, refs)
    if unverified:
        warnings.append("unverified_numbers")
    cited = list(dict.fromkeys(_CITATION.findall(answer)))
    if not cited:
        warnings.append("no_citations")
    if len(answer) > max_chars:
        answer = answer[:max_chars]
        warnings.append("truncated")
    return Validated(answer, cited, warnings, unverified)
