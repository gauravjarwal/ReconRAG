from __future__ import annotations

import logging
import threading

from chromadb import Collection

from config import settings
from models.schemas import SourceInfo
from pipelines.bm25 import Bm25Index, build_index, search_index
from pipelines.embedding import embed_query
from pipelines.fusion import reciprocal_rank_fusion
from pipelines.reranker import rerank
from pipelines.retry import UpstreamServiceError
from pipelines.usage import ActionUsage, timed
from pipelines.vector_store import collection_revision, get_all_chunks, query_collection

logger = logging.getLogger(__name__)

# collection name -> ((revision, chunk count), index). Rebuilt when either changes.
_bm25_cache: dict[str, tuple[tuple[int, int], Bm25Index]] = {}
_bm25_lock = threading.Lock()

NO_RERANK_SCORE = 7.0


def _get_bm25_index(collection: Collection) -> Bm25Index:
    key = (collection_revision(collection.name), collection.count())
    with _bm25_lock:
        cached = _bm25_cache.get(collection.name)
        if cached and cached[0] == key:
            return cached[1]
    index = build_index(get_all_chunks(collection))
    with _bm25_lock:
        _bm25_cache[collection.name] = (key, index)
    return index


def retrieve(
    collection: Collection,
    question: str,
    reranking: bool = True,
) -> tuple[list[SourceInfo], float, list[ActionUsage]]:
    """
    Full hybrid retrieval pipeline.

    Returns:
        (sources, best_score, action_usages)
        sources is empty if the confidence gate blocks.
        action_usages contains usage for: query_embedding, vector_search, bm25_search, reranking.

    Raises:
        UpstreamServiceError if the query cannot be embedded.
    """
    action_usages: list[ActionUsage] = []

    # 1. Embed query (returns usage)
    try:
        query_embedding, embed_usage = embed_query(question)
    except Exception as exc:
        logger.error("Query embedding failed: %s", type(exc).__name__)
        raise UpstreamServiceError("Embedding service unavailable.") from exc
    action_usages.append(embed_usage)

    # Decide fetch size: if reranking, fetch broad candidates; otherwise fetch only top 3
    fetch_k = settings.vector_top_k if reranking else settings.rerank_top_n
    bm25_k = settings.bm25_top_k if reranking else settings.rerank_top_n

    # 2. Vector search
    with timed() as ms:
        vector_results = query_collection(collection, query_embedding, n_results=fetch_k)
    action_usages.append(ActionUsage(action="vector_search", latency_ms=ms[0]))
    logger.debug("Vector search returned %d results", len(vector_results))

    # 3. BM25 search (index cached per collection revision)
    with timed() as ms:
        bm25_results = search_index(_get_bm25_index(collection), question, top_k=bm25_k)
    action_usages.append(ActionUsage(action="bm25_search", latency_ms=ms[0]))
    logger.debug("BM25 search returned %d results", len(bm25_results))

    # 4. RRF fusion
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    logger.debug("After RRF fusion: %d unique candidates", len(fused))

    if not fused:
        return [], 0.0, action_usages

    # 5. Rerank or direct top-N
    reranked = False
    if reranking:
        try:
            scored, rerank_usage = rerank(question, fused)
            action_usages.append(rerank_usage)
            top_n = scored[: settings.rerank_top_n]
            reranked = True
        except Exception as exc:
            # Degrade to RRF order rather than failing the whole query.
            logger.warning("Re-ranking failed (%s); falling back to RRF order", type(exc).__name__)
    if not reranked:
        top_n = [(chunk, NO_RERANK_SCORE) for chunk in fused[: settings.rerank_top_n]]
        logger.info("Using top %d RRF results without re-ranking", settings.rerank_top_n)

    # 6. Confidence gate (only meaningful with real re-rank scores)
    best_score = top_n[0][1] if top_n else 0.0
    if reranked and best_score < settings.confidence_threshold:
        logger.info("Confidence gate triggered (best score %.2f < %.2f)", best_score, settings.confidence_threshold)
        return [], best_score, action_usages

    sources = [
        SourceInfo(
            filename=chunk["metadata"].get("filename", "unknown"),
            chunk_index=int(chunk["metadata"].get("chunk_index", 0)),
            chunk_text=chunk["metadata"].get("chunk_text", chunk["document"]),
            score=score,
            injection_risk=bool(chunk["metadata"].get("injection_risk", False)),
        )
        for chunk, score in top_n
    ]

    return sources, best_score, action_usages
