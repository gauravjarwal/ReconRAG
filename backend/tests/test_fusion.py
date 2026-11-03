from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipelines.fusion import reciprocal_rank_fusion


def _make_chunk(doc_id: str) -> dict:
    return {"id": doc_id, "document": f"doc {doc_id}", "metadata": {}}


def test_empty_lists_return_empty():
    assert reciprocal_rank_fusion([], []) == []


def test_single_list_preserves_order():
    chunks = [_make_chunk(str(i)) for i in range(5)]
    result = reciprocal_rank_fusion(chunks, [])
    ids = [r["id"] for r in result]
    assert ids == [str(i) for i in range(5)]


def test_shared_docs_rank_higher():
    # doc "A" appears in rank 1 in both lists — should score highest
    vector = [_make_chunk("A"), _make_chunk("B"), _make_chunk("C")]
    bm25 = [_make_chunk("A"), _make_chunk("D"), _make_chunk("E")]
    result = reciprocal_rank_fusion(vector, bm25)
    assert result[0]["id"] == "A"


def test_deduplication():
    chunks = [_make_chunk("X")]
    result = reciprocal_rank_fusion(chunks, chunks)
    ids = [r["id"] for r in result]
    assert ids.count("X") == 1


def test_all_unique_docs_merged():
    vector = [_make_chunk("v1"), _make_chunk("v2")]
    bm25 = [_make_chunk("b1"), _make_chunk("b2")]
    result = reciprocal_rank_fusion(vector, bm25)
    assert len(result) == 4
    assert {r["id"] for r in result} == {"v1", "v2", "b1", "b2"}
