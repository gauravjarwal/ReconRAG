"""
PHI detection for the HIPAA Safe Harbor identifiers that can appear in text.

Two detectors, merged:
  1. Regex rules for structured identifiers (SSN, MRN, phone, dates, ...).
     Context-gated where the bare pattern would be too noisy (an MRN is just
     "some digits" without the word "MRN" next to it).
  2. spaCy NER for names and locations, which no regex can find reliably.

Detection is recall-oriented: over-redacting a cosmology paper costs answer
quality, under-redacting a discharge summary leaks PHI to third-party APIs.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhiSpan:
    start: int
    end: int
    entity_type: str
    text: str
    # Full name a partial mention refers to ("Gonzalez" -> "Maria Gonzalez"), so
    # both get the same token. None means the span stands for itself.
    canonical: str | None = None


@dataclass(frozen=True)
class _Rule:
    entity_type: str
    pattern: re.Pattern
    group: int = 0
    validator: Callable[[str], bool] | None = None


def _has_digit(value: str) -> bool:
    return any(c.isdigit() for c in value)


def _luhn_valid(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _valid_ipv4(value: str) -> bool:
    return all(0 <= int(octet) <= 255 for octet in value.split("."))


def _vin_like(value: str) -> bool:
    return _has_digit(value) and any(c.isalpha() for c in value)


_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_US_STATE = (
    r"(?:A[KLRZ]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|PA|RI|"
    r"S[CD]|T[NX]|UT|V[AT]|W[AIVY])"
)
_ID_VALUE = r"([A-Za-z0-9][A-Za-z0-9-]{3,24})\b"
_LABEL_SEP = r"\s*(?:number|no\.?|num|#)?\s*[:#]?\s*"
# A capitalised word that is not an all-caps label ("MRN", "DOB"). Name parts are
# joined by a single space so a match can't run across a column gap into the next field.
_NAME = r"(?![A-Z]{2,}\b)[A-Z][a-zA-Z'’-]+"

RULES: list[_Rule] = [
    # Identification numbers
    _Rule("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    _Rule("SSN", re.compile(r"(?i)\b(?:SSN|social\s+security)" + _LABEL_SEP + r"(\d{9}|\d{3} \d{2} \d{4})\b"), 1),
    _Rule("MRN", re.compile(r"(?i)\b(?:MRN|MR#|medical\s+record|chart)" + _LABEL_SEP + _ID_VALUE), 1, _has_digit),
    _Rule(
        "HEALTH_PLAN_ID",
        re.compile(
            r"(?i)\b(?:member|subscriber|policy|insurance|beneficiary|health\s+plan|group|medicaid|medicare)\s*"
            r"(?:id|number|no\.?|#)\s*[:#]?\s*" + _ID_VALUE
        ),
        1,
        _has_digit,
    ),
    # Medicare Beneficiary Identifier: fixed 11-character format
    _Rule(
        "HEALTH_PLAN_ID",
        re.compile(
            r"\b[1-9][AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d[AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d"
            r"[AC-HJKMNP-RT-Y]{2}\d{2}\b"
        ),
    ),
    _Rule("ACCOUNT_NUMBER", re.compile(r"(?i)\b(?:account|acct)\.?" + _LABEL_SEP + _ID_VALUE), 1, _has_digit),
    _Rule(
        "LICENSE_NUMBER",
        re.compile(r"(?i)\b(?:licen[cs]e|certificate|DEA|NPI)" + _LABEL_SEP + _ID_VALUE),
        1,
        _has_digit,
    ),
    _Rule(
        "DEVICE_ID",
        re.compile(r"(?i)\b(?:serial|device\s+id|UDI|implant\s+id|IMEI)" + _LABEL_SEP + _ID_VALUE),
        1,
        _has_digit,
    ),
    _Rule("VEHICLE_ID", re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"), 0, _vin_like),
    _Rule(
        "VEHICLE_ID",
        re.compile(r"(?i)\b(?:license\s+plate|plate)" + _LABEL_SEP + r"([A-Z0-9-]{2,8})\b"),
        1,
        _has_digit,
    ),
    _Rule("CREDIT_CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), 0, _luhn_valid),
    # Contact details
    _Rule("PHONE", re.compile(r"(?<![\w-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-])")),
    _Rule("PHONE", re.compile(r"(?i)\b(?:phone|tel|fax|cell|mobile|pager)" + _LABEL_SEP + r"(\+?\d{10,11})\b"), 1),
    _Rule("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    _Rule("URL", re.compile(r"\b(?:https?://|www\.)[^\s<>\"'\])]*[^\s<>\"'\]).,;:!?]")),
    _Rule("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0, _valid_ipv4),
    # Geography smaller than a state
    _Rule(
        "ADDRESS",
        re.compile(
            r"\b\d{1,6}\s+(?:[A-Z][a-z]+\.?\s+){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|"
            r"Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Ter|Circle|Cir|Highway|Hwy|Parkway|Pkwy)\b\.?"
            r"(?:,?\s+(?:Apt|Apartment|Suite|Ste|Unit|#)\.?\s*[A-Za-z0-9-]+)?"
        ),
    ),
    _Rule("ZIP", re.compile(r"\b" + _US_STATE + r",?\s+(\d{5}(?:-\d{4})?)\b"), 1),
    _Rule("ZIP", re.compile(r"(?i)\bzip(?:\s*code)?\s*[:#]?\s*(\d{5}(?:-\d{4})?)\b"), 1),
    _Rule("ZIP", re.compile(r"\b\d{5}-\d{4}\b")),
    # Dates: every element except the year is an identifier
    _Rule("DATE", re.compile(r"\b\d{1,2}([/-])\d{1,2}\1(?:\d{4}|\d{2})\b")),
    _Rule("DATE", re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b")),  # dotted needs a 4-digit year: "3.10.12" is a version
    _Rule("DATE", re.compile(r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b")),
    _Rule("DATE", re.compile(r"\b" + _MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b")),
    _Rule("DATE", re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH + r"\.?(?:,?\s+\d{4})?\b")),
    _Rule("DATE", re.compile(r"\b" + _MONTH + r"\.?,?\s+\d{4}\b")),
    # Ages over 89 must be aggregated; we pseudonymize the number
    _Rule("AGE", re.compile(r"(?i)(?<![\^.\d])\b(9\d|1[0-4]\d)\s*-?\s*(?:(?:years?|yrs?)[\s-]*old|y/?o)\b"), 1),
    _Rule("AGE", re.compile(r"(?i)\baged?\s*[:]?\s*(9\d|1[0-4]\d)\b"), 1),
    # Name backstops for when NER misses a labeled or titled name
    _Rule("PERSON", re.compile(r"\b(?:Patient(?:\s+name)?|Name|Pt|Guarantor|Emergency\s+contact)\s*:[ \t]*(" + _NAME + r"(?: " + _NAME + r"){0,3})"), 1),
    _Rule("PERSON", re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr)\.?[ \t]+(" + _NAME + r"(?: " + _NAME + r"){0,2})"), 1),
]

# NER label -> PHI entity type. ORG and DATE are deliberately excluded:
# organisations aren't Safe Harbor identifiers, and spaCy's DATE fires on
# "three days" and "2023", which are not PHI. Dates come from the regex rules.
_NER_LABELS = {"PERSON": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION"}

# Higher wins when two spans of equal length overlap.
_PRIORITY = {"PERSON": 1, "LOCATION": 1}

_NER_SEGMENT_CHARS = 100_000

_nlp = None


def load_ner(model_name: str) -> None:
    """Load the spaCy model once at startup. Raises if the model is missing (fail closed)."""
    global _nlp
    import spacy

    _nlp = spacy.load(model_name, exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "senter"])
    logger.info("Loaded NER model %s", model_name)


def ner_loaded() -> bool:
    return _nlp is not None


def _segments(text: str) -> Iterator[tuple[int, str]]:
    """Yield (offset, segment) pieces under the NER size cap, split on line breaks."""
    start = 0
    while start < len(text):
        end = min(start + _NER_SEGMENT_CHARS, len(text))
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start:
                end = newline + 1
        yield start, text[start:end]
        start = end


def _detect_ner(text: str) -> list[PhiSpan]:
    if _nlp is None:
        return []
    offsets, segments = zip(*_segments(text)) if text else ((), ())
    spans: list[PhiSpan] = []
    for offset, doc in zip(offsets, _nlp.pipe(segments)):
        for ent in doc.ents:
            entity_type = _NER_LABELS.get(ent.label_)
            if entity_type and len(ent.text.strip()) > 1:
                start = offset + ent.start_char
                spans.append(PhiSpan(start, start + len(ent.text), entity_type, ent.text))
    return spans


def detect_structured_phi(text: str) -> list[PhiSpan]:
    """Regex-only detection. Cheap enough for log lines and LLM output."""
    spans: list[PhiSpan] = []
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            value = match.group(rule.group)
            if not value or (rule.validator and not rule.validator(value)):
                continue
            spans.append(PhiSpan(match.start(rule.group), match.end(rule.group), rule.entity_type, value))
    return resolve_overlaps(spans)


def _propagate_names(text: str, spans: list[PhiSpan]) -> list[PhiSpan]:
    """
    NER is inconsistent within a document: it tags "John Smith" in one sentence
    and misses "Smith" or "john smith" three paragraphs later. Re-find every
    detected name (and each part of a multi-word name) everywhere in the text.
    """
    found: list[PhiSpan] = []
    names = {s.text for s in spans if s.entity_type == "PERSON"}
    for name in names:
        pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE)
        found.extend(PhiSpan(m.start(), m.end(), "PERSON", m.group(0)) for m in pattern.finditer(text))
        for part in name.split():
            if len(part) >= 3 and part[0].isupper():
                part_re = re.compile(r"(?<!\w)" + re.escape(part) + r"(?!\w)")
                found.extend(PhiSpan(m.start(), m.end(), "PERSON", m.group(0)) for m in part_re.finditer(text))
    return found


def detect_phi(text: str) -> list[PhiSpan]:
    """Full detection: regex + NER + in-document name propagation, overlaps resolved."""
    spans = detect_structured_phi(text) + _detect_ner(text)
    spans += _propagate_names(text, spans)
    return _link_partial_names(resolve_overlaps(spans))


def _link_partial_names(spans: list[PhiSpan]) -> list[PhiSpan]:
    """
    Point single-word name mentions at the one full name they belong to, so
    "Maria Gonzalez" and "Ms. Gonzalez" share a token and the LLM knows they're
    the same person. Ambiguous parts (two different Smiths) are left alone:
    a wrong link would be worse than no link.
    """
    owners: dict[str, set[str]] = {}
    for s in spans:
        if s.entity_type == "PERSON" and len(s.text.split()) > 1:
            full = " ".join(s.text.split())
            for part in full.split():
                owners.setdefault(part.casefold(), set()).add(full.casefold())
    full_names = {" ".join(s.text.split()).casefold(): " ".join(s.text.split()) for s in spans if s.entity_type == "PERSON"}
    linked: list[PhiSpan] = []
    for s in spans:
        candidates = owners.get(s.text.casefold(), set()) if s.entity_type == "PERSON" and len(s.text.split()) == 1 else set()
        linked.append(replace(s, canonical=full_names[next(iter(candidates))]) if len(candidates) == 1 else s)
    return linked


def resolve_overlaps(spans: list[PhiSpan]) -> list[PhiSpan]:
    """Sweep left to right; when spans overlap keep the longer one (ties: structured over NER)."""
    result: list[PhiSpan] = []
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if result and span.start < result[-1].end:
            current = result[-1]
            span_key = (span.end - span.start, -_PRIORITY.get(span.entity_type, 0))
            current_key = (current.end - current.start, -_PRIORITY.get(current.entity_type, 0))
            if span_key > current_key:
                result[-1] = span
            continue
        result.append(span)
    return result


def redact_structured_phi(text: str) -> str:
    """Replace structured PHI with a type label. Used by the log filter."""
    for span in reversed(detect_structured_phi(text)):
        text = text[: span.start] + f"[{span.entity_type}]" + text[span.end :]
    return text
