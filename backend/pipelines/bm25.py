from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


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
    if not chunks:
        return []

    corpus = [_tokenize(c["document"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    scored = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    return [chunk for _, chunk in scored[:top_k]]
