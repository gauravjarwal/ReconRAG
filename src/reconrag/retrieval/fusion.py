"""Merges ranked lists, and drops identical text from the merged list."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from reconrag.domain import Hit


def weighted_rrf(lists: Sequence[tuple[Sequence[Hit], float]], k: int) -> list[Hit]:
    """score = sum of weight / (k + rank). Uses ranks, so incomparable scores can be combined."""
    merged: dict[UUID, Hit] = {}
    for hits, weight in lists:
        for rank, hit in enumerate(hits, start=1):
            kept = merged.setdefault(hit.chunk_id, hit)
            kept.fused_score += weight / (k + rank)
            if kept is not hit:
                kept.dense_similarity = kept.dense_similarity or hit.dense_similarity
                kept.lexical_score = kept.lexical_score or hit.lexical_score
    return sorted(merged.values(), key=lambda hit: hit.fused_score, reverse=True)


def suppress_identical(hits: Sequence[Hit]) -> list[Hit]:
    """Boilerplate shared by many documents would otherwise fill every slot."""
    kept: dict[bytes, Hit] = {}
    for hit in hits:
        first = kept.setdefault(hit.text_hash, hit)
        if first is not hit:
            first.signals.setdefault("also_in", []).append(hit.filename)
    return list(kept.values())
