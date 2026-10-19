"""
Keyed, deterministic pseudonymization of PHI.

Every detected identifier is replaced with a token like [PERSON_3f2a9c1b0d4e],
where the hex part is HMAC-SHA256(secret, collection | type | normalized value).

Why keyed and deterministic:
  - Deterministic: "John Smith" maps to the same token in every chunk and in
    the user's query, so BM25 and the LLM can still connect mentions of the
    same person without ever seeing the name.
  - Keyed: a plain hash of an SSN is reversible by brute force (10^9 values).
    Without the HMAC key, tokens reveal nothing.
  - Collection-scoped: the same person gets unrelated tokens in two
    collections, so tokens can't be used to link records across them.

The original values go to the encrypted vault so authorised callers can
re-identify answers. Nothing downstream of this module (embeddings, BM25,
re-ranker, generator, Chroma) ever sees raw PHI.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from config import settings
from pipelines.phi import PhiSpan, detect_phi, resolve_overlaps
from pipelines.phi_vault import PhiVault

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\[([A-Z][A-Z_]*)_([0-9a-fA-F]{12})\]")

_DIGIT_TYPES = {"SSN", "PHONE", "CREDIT_CARD", "ZIP"}
_ID_TYPES = {"MRN", "HEALTH_PLAN_ID", "ACCOUNT_NUMBER", "LICENSE_NUMBER", "DEVICE_ID", "VEHICLE_ID"}
# Free-text types a query might mention without NER noticing (lowercase, partial).
_DICTIONARY_TYPES = ("PERSON", "LOCATION")
_MAX_NGRAM = 4

_vault: PhiVault | None = None


def init_vault() -> None:
    global _vault
    _vault = PhiVault(settings.phi_vault_path, settings.phi_encryption_key)


def get_vault() -> PhiVault:
    if _vault is None:
        init_vault()
    return _vault


@dataclass
class PseudonymizationResult:
    text: str
    entity_counts: Counter = field(default_factory=Counter)


def _normalize(entity_type: str, value: str) -> str:
    if entity_type in _DIGIT_TYPES:
        return re.sub(r"\D", "", value)
    if entity_type in _ID_TYPES:
        return re.sub(r"[\s-]", "", value).upper()
    return " ".join(value.casefold().split())


def make_token(collection: str, entity_type: str, value: str) -> str:
    message = f"{collection}\x1f{entity_type}\x1f{_normalize(entity_type, value)}".encode()
    digest = hmac.new(settings.phi_hmac_key.encode(), message, hashlib.sha256).hexdigest()
    return f"[{entity_type}_{digest[:12]}]"


def _replace_spans(
    collection: str, text: str, spans: list[PhiSpan], known_aliases: dict[str, str] | None = None
) -> PseudonymizationResult:
    """
    Swap each span for its token and record token -> value in the vault.
    A linked partial name ("Gonzalez") gets its full name's token, and the
    partial's own token is recorded as an alias so a later query that only
    says "Gonzalez" can be rewritten to the same token.
    """
    known_aliases = known_aliases or {}
    entries: dict[str, tuple[str, str]] = {}
    aliases: dict[str, str] = {}
    counts: Counter = Counter()
    parts: list[str] = []
    cursor = 0
    for span in spans:
        value = span.canonical or span.text
        token = make_token(collection, span.entity_type, value)
        if span.canonical:
            aliases[make_token(collection, span.entity_type, span.text)] = token
        if token in known_aliases:
            token = known_aliases[token]
        else:
            entries.setdefault(token, (span.entity_type, value))
        counts[span.entity_type] += 1
        parts.append(text[cursor : span.start])
        parts.append(token)
        cursor = span.end
    parts.append(text[cursor:])
    get_vault().store(collection, entries)
    get_vault().store_aliases(collection, aliases)
    return PseudonymizationResult("".join(parts), counts)


def pseudonymize_document(collection: str, text: str) -> PseudonymizationResult:
    """De-identify a whole document before chunking, so NER sees full context."""
    return _replace_spans(collection, text, detect_phi(text))


def _dictionary_spans(collection: str, text: str, taken: list[PhiSpan]) -> list[PhiSpan]:
    """
    Find word n-grams whose token already exists in the collection's vault.

    Catches "what meds is john smith on" when NER misses the lowercase name:
    we can't search encrypted values, but we can recompute the HMAC token for
    each candidate n-gram and ask the vault whether that token exists.
    """
    words = list(re.finditer(r"[\w'’-]+", text))
    candidates: dict[str, PhiSpan] = {}
    for i in range(len(words)):
        for n in range(1, _MAX_NGRAM + 1):
            if i + n > len(words):
                break
            start, end = words[i].start(), words[i + n - 1].end()
            if any(start < s.end and s.start < end for s in taken):
                continue
            value = text[start:end]
            for entity_type in _DICTIONARY_TYPES:
                candidates[make_token(collection, entity_type, value)] = PhiSpan(start, end, entity_type, value)
    if not candidates:
        return []
    known = get_vault().existing(collection, list(candidates))
    return [span for token, span in candidates.items() if token in known]


def pseudonymize_query(collection: str, query: str) -> PseudonymizationResult:
    spans = detect_phi(query)
    spans = resolve_overlaps(spans + _dictionary_spans(collection, query, spans))
    # "What is Gonzalez allergic to?": the documents used Maria Gonzalez's token
    # for every mention, so map the surname's token onto it.
    candidate_tokens = [make_token(collection, s.entity_type, s.canonical or s.text) for s in spans]
    return _replace_spans(collection, query, spans, get_vault().resolve_aliases(collection, candidate_tokens))


def reidentify(collection: str, text: str) -> str:
    """Swap tokens back to original values. Unknown tokens are left as-is."""
    matches = list(TOKEN_PATTERN.finditer(text))
    if not matches:
        return text
    canonical = {m.group(0): f"[{m.group(1)}_{m.group(2).lower()}]" for m in matches}
    originals = get_vault().resolve(collection, list(set(canonical.values())))
    return TOKEN_PATTERN.sub(lambda m: originals.get(canonical[m.group(0)], m.group(0)), text)


def contains_tokens(text: str) -> bool:
    return TOKEN_PATTERN.search(text) is not None
