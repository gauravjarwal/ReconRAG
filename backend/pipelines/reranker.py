from __future__ import annotations

import json
import logging

import google.generativeai as genai

from config import settings
from pipelines.usage import ActionUsage, cost_gemini_flash, timed

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.google_api_key, transport="rest")


def rerank(query: str, candidates: list[dict]) -> tuple[list[tuple[dict, float]], ActionUsage]:
    """
    Rerank candidate chunks using a single Gemini LLM call.

    Sends query + all chunks in one call, receives JSON array of scores (1-10).
    Token counts are read from Gemini's response.usage_metadata.

    Returns:
        (scored_candidates, ActionUsage)
        scored_candidates: list of (chunk, score) sorted by score descending.
    """
    if not candidates:
        return [], ActionUsage(action="reranking")

    numbered_chunks = "\n\n".join(
        f"[{i}] {c['document'][:400]}" for i, c in enumerate(candidates)
    )

    prompt = f"""You are a relevance scoring assistant.

Query: {query}

Rate how relevant each of the following document chunks is to answering the query.
Return ONLY a JSON array with objects containing "index" (int) and "score" (float 1-10).
Higher score = more relevant. No explanation, just the JSON array.

Chunks:
{numbered_chunks}

Response format: [{{"index": 0, "score": 7.5}}, {{"index": 1, "score": 3.0}}, ...]"""

    model = genai.GenerativeModel(settings.gemini_generation_model)

    with timed() as ms:
        response = model.generate_content(prompt)

    # Extract token counts from Gemini usage_metadata
    meta = getattr(response, "usage_metadata", None)
    tokens_in = getattr(meta, "prompt_token_count", 0) or 0
    tokens_out = getattr(meta, "candidates_token_count", 0) or 0
    total_tokens = tokens_in + tokens_out

    usage = ActionUsage(
        action="reranking",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=total_tokens,
        cost_usd=cost_gemini_flash(tokens_in, tokens_out),
        latency_ms=ms[0],
    )

    raw = response.text.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        logger.warning("Reranker returned invalid JSON; assigning uniform scores")
        return [(c, 5.0) for c in candidates], usage

    try:
        scores_data: list[dict] = json.loads(raw[start:end])
    except json.JSONDecodeError:
        logger.warning("Failed to parse reranker JSON; assigning uniform scores")
        return [(c, 5.0) for c in candidates], usage

    score_map: dict[int, float] = {item["index"]: float(item["score"]) for item in scores_data}
    scored = [
        (candidates[i], score_map.get(i, 0.0))
        for i in range(len(candidates))
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True), usage
