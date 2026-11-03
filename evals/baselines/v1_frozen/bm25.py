from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
class Bm25Index:
    chunks: list[dict]
    bm25: BM25Okapi | None


def build_index(chunks: list[dict]) -> Bm25Index:
    """Tokenize and index chunks once; reuse the index across queries."""
    if not chunks:
        return Bm25Index(chunks=[], bm25=None)
    return Bm25Index(chunks=chunks, bm25=BM25Okapi([_tokenize(c["document"]) for c in chunks]))


def search_index(index: Bm25Index, query: str, top_k: int) -> list[dict]:
    if index.bm25 is None:
        return []
    scores = index.bm25.get_scores(_tokenize(query))
    scored = sorted(zip(scores, index.chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def bm25_search(chunks: list[dict], query: str, top_k: int) -> list[dict]:
    """
    Build a BM25 index from chunks and return the top_k results.

    Args:
        chunks: list of dicts with at least 'id' and 'document' keys
        query: search query string
        top_k: number of results to return

    Returns:
        List of chunk dicts sorted by BM25 score descending, limited to top_k.
    """
    return search_index(build_index(chunks), query, top_k)
