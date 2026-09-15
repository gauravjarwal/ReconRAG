from __future__ import annotations

RRF_K = 60


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    score(d) = sum(1 / (k + rank(d))) over each list that contains d.
    Documents are identified by their 'id' field.

    Returns:
        List of unique chunk dicts sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    id_to_chunk: dict[str, dict] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        doc_id = chunk["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        id_to_chunk[doc_id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        doc_id = chunk["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        id_to_chunk[doc_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
    return [id_to_chunk[doc_id] for doc_id in sorted_ids]
