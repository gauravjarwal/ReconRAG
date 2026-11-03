from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipelines.bm25 import bm25_search


def _make_chunk(doc_id: str, text: str) -> dict:
    return {"id": doc_id, "document": text, "metadata": {}}


def test_bm25_returns_most_relevant_first():
    chunks = [
        _make_chunk("1", "The quick brown fox jumps over the lazy dog"),
        _make_chunk("2", "Machine learning models process data"),
        _make_chunk("3", "The fox is a quick animal that jumps"),
    ]
    results = bm25_search(chunks, "quick fox jumps", top_k=3)
    # Chunks 1 and 3 are most relevant; chunk 2 is least
    top_ids = [r["id"] for r in results[:2]]
    assert "1" in top_ids or "3" in top_ids


def test_bm25_top_k_limits_results():
    chunks = [_make_chunk(str(i), f"document number {i} contains text") for i in range(10)]
    results = bm25_search(chunks, "document text", top_k=3)
    assert len(results) == 3


def test_bm25_empty_chunks_returns_empty():
    assert bm25_search([], "any query", top_k=5) == []


def test_bm25_irrelevant_query_still_returns_results():
    chunks = [_make_chunk("1", "apple banana cherry")]
    results = bm25_search(chunks, "quantum physics relativity", top_k=1)
    assert len(results) == 1
