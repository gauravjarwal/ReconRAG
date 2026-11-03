"""The three searches. Each returns ranked hits for one collection's active chunks."""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

import numpy as np
from pgvector import HalfVector

from reconrag.config import Settings
from reconrag.domain import Hit
from reconrag.storage.db import Conn

_COLUMNS = """c.id, c.version_id, c.document_id, c.ordinal, c.text, c.context, c.section_path,
    c.page_start, c.page_end, c.kind, c.text_hash, c.injection_risk, v.filename"""

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/.]*[A-Za-z0-9]|[A-Za-z0-9]")
_FUZZY_MIN_LENGTH = 5
_FUZZY_THRESHOLD = 0.7


def identifiers(query: str) -> list[str]:
    """Codes and names a stemmer would damage: AC-200, CF7731, HIPAA."""
    found = []
    for token in _TOKEN.findall(query):
        has_digit = any(c.isdigit() for c in token) and any(c.isalpha() for c in token)
        if has_digit or "-" in token or (token.isupper() and len(token) >= 3):
            found.append(token)
    return list(dict.fromkeys(found))


def _hit(row: dict[str, Any], **scores: float | None) -> Hit:
    return Hit(
        chunk_id=row["id"],
        version_id=row["version_id"],
        document_id=row["document_id"],
        filename=row["filename"],
        ordinal=row["ordinal"],
        text=row["text"],
        context=row["context"],
        section_path=row["section_path"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        kind=row["kind"],
        text_hash=bytes(row["text_hash"]),
        injection_risk=row["injection_risk"],
        **scores,  # type: ignore[arg-type]
    )


def dense_search(
    conn: Conn, collection_id: UUID, vector: np.ndarray, k: int, settings: Settings
) -> list[Hit]:
    """HNSW filters after the scan, so iterative scans keep going until k rows pass the filter."""
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(settings.hnsw_ef_search)}")
        conn.execute(f"SET LOCAL hnsw.max_scan_tuples = {int(settings.hnsw_max_scan_tuples)}")
        rows = conn.execute(
            f"""
            SELECT {_COLUMNS}, c.embedding <=> %(vector)s AS distance
            FROM chunks c JOIN document_versions v ON v.id = c.version_id
            WHERE c.active AND c.collection_id = %(collection)s
            ORDER BY c.embedding <=> %(vector)s LIMIT %(k)s
            """,
            {"vector": HalfVector(vector), "collection": collection_id, "k": k},
        ).fetchall()
    # Relaxed order can be slightly out of order, so sort the final page exactly.
    hits = [_hit(row, dense_similarity=1.0 - float(row["distance"])) for row in rows]
    return sorted(hits, key=lambda hit: -(hit.dense_similarity or 0.0))


def lexical_search(conn: Conn, collection_id: UUID, query: str, k: int) -> list[Hit]:
    """OR, not AND: a question's words rarely all appear in the one passage that answers it."""
    tokens = identifiers(query)
    rows = conn.execute(
        f"""
        WITH q AS (
            SELECT replace(plainto_tsquery('english', %(query)s)::text, '&', '|')::tsquery
                   || coalesce(
                       (SELECT tsquery_or FROM (
                           SELECT string_agg(quote_literal(t), ' | ')::tsquery AS tsquery_or
                           FROM unnest(%(tokens)s::text[]) AS t) s),
                       ''::tsquery) AS query
        )
        SELECT {_COLUMNS}, ts_rank(c.search, q.query, 1) AS rank
        FROM chunks c JOIN document_versions v ON v.id = c.version_id, q
        WHERE c.active AND c.collection_id = %(collection)s AND c.search @@ q.query
        ORDER BY rank DESC LIMIT %(k)s
        """,
        {
            "query": query,
            "tokens": [t.casefold() for t in tokens],
            "collection": collection_id,
            "k": k,
        },
    ).fetchall()
    return [_hit(row, lexical_score=float(row["rank"])) for row in rows]


def fuzzy_search(conn: Conn, collection_id: UUID, query: str, k: int) -> list[Hit]:
    """Trigram match on identifiers only, for OCR slips like 'CF-773l'. Too slow for whole text."""
    tokens = [t for t in identifiers(query) if len(t) >= _FUZZY_MIN_LENGTH]
    if not tokens:
        return []
    with conn.transaction():
        conn.execute(f"SET LOCAL pg_trgm.word_similarity_threshold = {_FUZZY_THRESHOLD}")
        rows = conn.execute(
            f"""
            SELECT {_COLUMNS}, word_similarity(%(token)s, c.text) AS similarity
            FROM chunks c JOIN document_versions v ON v.id = c.version_id
            WHERE c.active AND c.collection_id = %(collection)s AND %(token)s <%% c.text
            ORDER BY similarity DESC LIMIT %(k)s
            """,
            {"token": " ".join(tokens[:3]), "collection": collection_id, "k": k},
        ).fetchall()
    return [_hit(row, lexical_score=float(row["similarity"])) for row in rows]
