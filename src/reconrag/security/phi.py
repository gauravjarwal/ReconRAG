"""Keyed, deterministic pseudonymisation.

Each identifier becomes a token like [PERSON_3f2a9c1b0d4e], the HMAC of (collection, type, value).
Deterministic, so the same person matches across chunks and queries. Keyed, because a plain hash of
an SSN can be brute-forced. Collection-scoped, so tokens can't link records across collections.
Nothing after this module (embeddings, search, the model) sees raw PHI.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from uuid import UUID

from reconrag.domain import Block, ParsedDocument
from reconrag.security.phi_detect import PhiSpan, detect_phi, resolve_overlaps
from reconrag.security.vault import PhiVault
from reconrag.storage.db import Conn

TOKEN_PATTERN = re.compile(r"\[([A-Z][A-Z_]*)_([0-9a-fA-F]{12})\]")
_DIGIT_TYPES = {"SSN", "PHONE", "CREDIT_CARD", "ZIP"}
_ID_TYPES = {
    "MRN",
    "HEALTH_PLAN_ID",
    "ACCOUNT_NUMBER",
    "LICENSE_NUMBER",
    "DEVICE_ID",
    "VEHICLE_ID",
}
_DICTIONARY_TYPES = ("PERSON", "LOCATION")
_MAX_NGRAM = 4


@dataclass
class Pseudonymized:
    text: str
    entity_counts: Counter[str] = field(default_factory=Counter)


def _normalize(entity_type: str, value: str) -> str:
    if entity_type in _DIGIT_TYPES:
        return re.sub(r"\D", "", value)
    if entity_type in _ID_TYPES:
        return re.sub(r"[\s-]", "", value).upper()
    return " ".join(value.casefold().split())


class Pseudonymizer:
    def __init__(self, hmac_key: str, vault: PhiVault) -> None:
        self._key = hmac_key.encode()
        self.vault = vault

    def token(self, collection_id: UUID, entity_type: str, value: str) -> str:
        message = f"{collection_id}\x1f{entity_type}\x1f{_normalize(entity_type, value)}"
        digest = hmac.new(self._key, message.encode(), hashlib.sha256).hexdigest()
        return f"[{entity_type}_{digest[:12]}]"

    def _replace(
        self,
        conn: Conn,
        collection_id: UUID,
        text: str,
        spans: list[PhiSpan],
        known_aliases: dict[str, str] | None = None,
    ) -> Pseudonymized:
        known_aliases = known_aliases or {}
        entries: dict[str, tuple[str, str]] = {}
        aliases: dict[str, str] = {}
        counts: Counter[str] = Counter()
        parts: list[str] = []
        cursor = 0
        for span in spans:
            value = span.canonical or span.text
            token = self.token(collection_id, span.entity_type, value)
            if span.canonical:
                aliases[self.token(collection_id, span.entity_type, span.text)] = token
            if token in known_aliases:
                token = known_aliases[token]
            else:
                entries.setdefault(token, (span.entity_type, value))
            counts[span.entity_type] += 1
            parts.extend((text[cursor : span.start], token))
            cursor = span.end
        parts.append(text[cursor:])
        self.vault.store(conn, collection_id, entries)
        self.vault.store_aliases(conn, collection_id, aliases)
        return Pseudonymized("".join(parts), counts)

    def document(
        self, conn: Conn, collection_id: UUID, document: ParsedDocument
    ) -> tuple[ParsedDocument, Counter[str]]:
        """Detects over the whole document, so NER has context, then rewrites block by block."""
        separator = "\n"
        fields: list[tuple[Block, tuple[int, int] | None]] = []
        texts: list[str] = []
        for block in document.blocks:
            if block.rows is None:
                fields.append((block, None))
                texts.append(block.text)
                continue
            for row_index, row in enumerate(block.rows):
                for cell_index, cell in enumerate(row):
                    fields.append((block, (row_index, cell_index)))
                    texts.append(cell)
        joined = separator.join(texts)
        spans = detect_phi(joined)
        pieces, counts = self._replace_fields(conn, collection_id, texts, separator, spans)
        for (block, position), new_text in zip(fields, pieces, strict=True):
            if position is None:
                block.text = new_text
            else:
                assert block.rows is not None
                block.rows[position[0]][position[1]] = new_text
        return document, counts

    def _replace_fields(
        self,
        conn: Conn,
        collection_id: UUID,
        texts: list[str],
        separator: str,
        spans: list[PhiSpan],
    ) -> tuple[list[str], Counter[str]]:
        """Spans are clipped to field bounds, so a match can't merge two cells."""
        pieces: list[str] = []
        counts: Counter[str] = Counter()
        offset = 0
        for text in texts:
            end = offset + len(text)
            local = [
                replace(
                    span, start=max(span.start, offset) - offset, end=min(span.end, end) - offset
                )
                for span in spans
                if span.start < end and span.end > offset
            ]
            result = self._replace(conn, collection_id, text, local)
            pieces.append(result.text)
            counts.update(result.entity_counts)
            offset = end + len(separator)
        return pieces, counts

    def _dictionary_spans(
        self, conn: Conn, collection_id: UUID, text: str, taken: list[PhiSpan]
    ) -> list[PhiSpan]:
        """Finds names NER misses in a query ("what is john smith on") by recomputing tokens."""
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
                for kind in _DICTIONARY_TYPES:
                    span = PhiSpan(start, end, kind, value)
                    candidates[self.token(collection_id, kind, value)] = span
        if not candidates:
            return []
        known = self.vault.known(conn, collection_id, list(candidates))
        return [span for token, span in candidates.items() if token in known]

    def query(self, conn: Conn, collection_id: UUID, query: str) -> Pseudonymized:
        spans = detect_phi(query)
        spans = resolve_overlaps(spans + self._dictionary_spans(conn, collection_id, query, spans))
        tokens = [self.token(collection_id, s.entity_type, s.canonical or s.text) for s in spans]
        aliases = self.vault.aliases(conn, collection_id, tokens)
        return self._replace(conn, collection_id, query, spans, aliases)

    def reidentify(self, conn: Conn, collection_id: UUID, text: str) -> str:
        matches = list(TOKEN_PATTERN.finditer(text))
        if not matches:
            return text
        canonical = {m.group(0): f"[{m.group(1)}_{m.group(2).lower()}]" for m in matches}
        originals = self.vault.resolve(conn, collection_id, list(set(canonical.values())))
        return TOKEN_PATTERN.sub(lambda m: originals.get(canonical[m.group(0)], m.group(0)), text)
