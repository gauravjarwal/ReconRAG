"""Decides whether retrieval found anything worth answering from.

Refusing is the point: a weak passage plus a capable model gives a confident wrong answer.
Works with or without a reranker. Thresholds are settings, to be calibrated on a dev set.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from reconrag.config import Settings
from reconrag.domain import Hit

_WORD = re.compile(r"[^\W_]{3,}")
_STOPWORDS = frozenset(
    "the and for are was were what which who whom how why when where does did can could should "
    "would will with from that this these those have has had not but any all about into your "
    "you our their there than then them they his her its".split()
)
# A question whose content words mostly appear in a passage is answered there, whatever the vector.
_MIN_TERM_COVERAGE = 0.6


@dataclass(frozen=True)
class GateDecision:
    refuse: bool
    reason: str
    confidence: float


def term_coverage(query: str, hit: Hit) -> float:
    terms = {w for w in _WORD.findall(query.casefold()) if w not in _STOPWORDS}
    if not terms:
        return 0.0
    haystack = f"{hit.context} {hit.text}".casefold()
    return sum(term in haystack for term in terms) / len(terms)


def decide(query: str, hits: Sequence[Hit], settings: Settings) -> GateDecision:
    if not hits:
        return GateDecision(True, "no_candidates", 0.0)
    best = hits[0]
    if best.rerank_score is not None:
        passed = best.rerank_score >= settings.gate_min_rerank
        return GateDecision(not passed, "ok" if passed else "low_rerank_score", best.rerank_score)
    dense = max((hit.dense_similarity or 0.0) for hit in hits)
    coverage = max(term_coverage(query, hit) for hit in hits)
    passed = dense >= settings.gate_min_dense or coverage >= _MIN_TERM_COVERAGE
    return GateDecision(not passed, "ok" if passed else "low_similarity", dense)
