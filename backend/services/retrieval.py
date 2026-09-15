from __future__ import annotations

import logging

from config import settings
from models.schemas import SourceInfo
from pipelines.bm25 import bm25_search
from pipelines.embedding import embed_query
from pipelines.fusion import reciprocal_rank_fusion
from pipelines.reranker import rerank
from pipelines.usage import ActionUsage, timed
from pipelines.vector_store import get_all_chunks, query_collection

logger = logging.getLogger(__name__)


def retrieve(
    collection: str,
    question: str,
    reranking: bool = True,
) -> tuple[list[SourceInfo], float, list[ActionUsage]]:
    """
    Full hybrid retrieval pipeline.

    Returns:
        (sources, best_score, action_usages)
        sources is empty and best_score is 0.0 if confidence gate blocks.
        action_usages contains usage for: query_embedding, vector_search, bm25_search, reranking.
    """
    action_usages: list[ActionUsage] = []

    # 1. Embed query (returns usage)
    query_embedding, embed_usage = embed_query(question)
    action_usages.append(embed_usage)

    # Decide fetch size: if reranking, fetch broad candidates; otherwise fetch only top 3
    fetch_k = settings.vector_top_k if reranking else settings.rerank_top_n
    bm25_k = settings.bm25_top_k if reranking else settings.rerank_top_n

    # 2. Vector search
    with timed() as ms:
        vector_results = query_collection(collection, query_embedding, n_results=fetch_k)
    action_usages.append(ActionUsage(action="vector_search", latency_ms=ms[0]))
    logger.debug("Vector search returned %d results", len(vector_results))

    # 3. BM25 search
    with timed() as ms:
        all_chunks = get_all_chunks(collection)
        bm25_results = bm25_search(all_chunks, question, top_k=bm25_k)
    action_usages.append(ActionUsage(action="bm25_search", latency_ms=ms[0]))
    logger.debug("BM25 search returned %d results", len(bm25_results))

    # 4. RRF fusion
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    logger.debug("After RRF fusion: %d unique candidates", len(fused))

    if not fused:
        return [], 0.0, action_usages

    # 5. Rerank or direct top-N
    if reranking:
        scored, rerank_usage = rerank(question, fused)
        action_usages.append(rerank_usage)
        top_n = scored[: settings.rerank_top_n]
    else:
        # No reranking — take top 3 directly from RRF, assign uniform score
        top_n = [(chunk, 7.0) for chunk in fused[: settings.rerank_top_n]]
        logger.info("Reranking disabled — using top %d RRF results directly", settings.rerank_top_n)

    # 6. Confidence gate (skip when reranking is off — trust RRF results)
    best_score = top_n[0][1] if top_n else 0.0
    if reranking and best_score < settings.confidence_threshold:
        logger.info("Confidence gate triggered (best score %.2f < %.2f)", best_score, settings.confidence_threshold)
        return [], best_score, action_usages

    sources = [
        SourceInfo(
            filename=chunk["metadata"].get("filename", "unknown"),
            chunk_index=int(chunk["metadata"].get("chunk_index", 0)),
            chunk_text=chunk["metadata"].get("chunk_text", chunk["document"]),
            score=score,
        )
        for chunk, score in top_n
    ]

    return sources, best_score, action_usages
