"""Builds what the model reads: the best passages, with neighbours, within a token budget."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from reconrag.domain import Hit
from reconrag.ingestion.chunking import count_tokens
from reconrag.storage.db import Conn


@dataclass(frozen=True)
class Passage:
    ref: str
    hit: Hit
    text: str


def neighbours(conn: Conn, hits: Sequence[Hit], reach: int) -> dict[UUID, list[dict]]:
    """Adjacent chunks of the same section, keyed by the hit's chunk id."""
    if not hits or reach <= 0:
        return {}
    rows = conn.execute(
        """
        SELECT h.id AS hit_id, c.ordinal, c.text
        FROM chunks h JOIN chunks c ON c.version_id = h.version_id
             AND c.ordinal BETWEEN h.ordinal - %(reach)s AND h.ordinal + %(reach)s
             AND c.ordinal <> h.ordinal AND c.section_path = h.section_path AND c.kind = 'text'
        WHERE h.id = ANY(%(ids)s) AND h.kind = 'text'
        ORDER BY h.id, c.ordinal
        """,
        {"ids": [hit.chunk_id for hit in hits], "reach": reach},
    ).fetchall()
    grouped: dict[UUID, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["hit_id"], []).append(row)
    return grouped


def _overlap_free(left: str, right: str) -> str:
    """Chunks overlap by design, so the shared sentences are removed when they're rejoined."""
    limit = min(len(left), len(right))
    for size in range(limit, 19, -1):
        if left.endswith(right[:size]):
            return left + right[size:]
    return f"{left} {right}"


def assemble(hits: Sequence[Hit], around: dict[UUID, list[dict]], max_tokens: int) -> list[Passage]:
    """Every hit gets a place before any neighbour, so expansion can't crowd out a better hit."""
    remaining = max_tokens
    chosen: list[Hit] = []
    for hit in hits:
        cost = count_tokens(hit.text)
        if cost > remaining and chosen:
            break
        chosen.append(hit)
        remaining -= cost
    used = {(hit.version_id, hit.ordinal) for hit in chosen}
    passages: list[Passage] = []
    for index, hit in enumerate(chosen, start=1):
        text = hit.text
        for row in around.get(hit.chunk_id, []):
            cost = count_tokens(row["text"])
            if (hit.version_id, row["ordinal"]) in used or cost > remaining:
                continue
            used.add((hit.version_id, row["ordinal"]))
            remaining -= cost
            text = (
                _overlap_free(row["text"], text)
                if row["ordinal"] < hit.ordinal
                else _overlap_free(text, row["text"])
            )
        passages.append(Passage(f"S{index}", hit, text))
    return passages
