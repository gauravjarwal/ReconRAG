from __future__ import annotations

import logging

import google.generativeai as genai

from config import settings
from pipelines.usage import ActionUsage, cost_gemini_embedding, estimate_tokens, timed

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.google_api_key, transport="rest")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Gemini gemini-embedding-001, batched. (Ingestion path — no usage tracking.)"""
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    batch_size = settings.embedding_batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = genai.embed_content(
            model=settings.gemini_embedding_model,
            content=batch,
            task_type="retrieval_document",
        )
        all_embeddings.extend(result["embedding"])

    logger.debug("Embedded %d texts in %d batches", len(texts), (len(texts) + batch_size - 1) // batch_size)
    return all_embeddings


def embed_query(text: str) -> tuple[list[float], ActionUsage]:
    """
    Embed a single query string and return usage metrics.

    Returns:
        (embedding_vector, ActionUsage)

    Note: Gemini embedding API does not return token counts.
    Tokens are estimated at 1 token per 4 characters.
    Cost uses character-based pricing ($0.00001 / 1K chars).
    """
    with timed() as ms:
        result = genai.embed_content(
            model=settings.gemini_embedding_model,
            content=text,
            task_type="retrieval_query",
        )

    estimated_tokens = estimate_tokens(text)
    usage = ActionUsage(
        action="query_embedding",
        tokens_in=estimated_tokens,
        tokens_out=0,
        total_tokens=estimated_tokens,
        cost_usd=cost_gemini_embedding(len(text)),
        latency_ms=ms[0],
    )
    return result["embedding"], usage
