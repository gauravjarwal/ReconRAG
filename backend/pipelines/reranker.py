from __future__ import annotations

import json
import logging

import google.generativeai as genai

from config import settings
from pipelines.chunking import CHUNK_SIZE
from pipelines.prompts import RERANK_PROMPT_V3
from pipelines.retry import with_retries
from pipelines.security import escape_prompt_tags
from pipelines.usage import ActionUsage, cost_gemini_flash, timed

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.google_api_key, transport="rest")


def _parse_scores(raw: str, n_candidates: int) -> dict[int, float] | None:
    """Parse the model's JSON array into {index: score}, ignoring malformed items. None if unusable."""
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return None
    try:
        items = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None

    scores: dict[int, float] = {}
    for item in items:
        try:
            index, score = int(item["index"]), float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= index < n_candidates:
            scores[index] = min(10.0, max(1.0, score))
    return scores or None


def rerank(query: str, candidates: list[dict]) -> tuple[list[tuple[dict, float]], ActionUsage]:
    """
    Rerank candidate chunks using a single Gemini LLM call.

    Sends query + all chunks in one call, receives JSON array of scores (1-10).
    Token counts are read from Gemini's response.usage_metadata.

    Raises on API failure; the caller decides how to degrade.

    Returns:
        (scored_candidates, ActionUsage)
        scored_candidates: list of (chunk, score) sorted by score descending.
    """
    if not candidates:
        return [], ActionUsage(action="reranking")

    numbered_chunks = "\n\n".join(
        # Whole chunk: a shorter cut hid the answer sentence from the scorer,
        # especially once PHI tokens (~22 chars each) lengthen the text.
        f"[{i}] {escape_prompt_tags(c['document'][:CHUNK_SIZE])}" for i, c in enumerate(candidates)
    )
    prompt = RERANK_PROMPT_V3.format(query=escape_prompt_tags(query), chunks=numbered_chunks)

    model = genai.GenerativeModel(settings.gemini_generation_model)

    with timed() as ms:
        response = with_retries(
            lambda: model.generate_content(prompt, request_options={"timeout": settings.llm_timeout_seconds}),
            operation="rerank",
        )

    # Extract token counts from Gemini usage_metadata
    meta = getattr(response, "usage_metadata", None)
    tokens_in = getattr(meta, "prompt_token_count", 0) or 0
    tokens_out = getattr(meta, "candidates_token_count", 0) or 0

    usage = ActionUsage(
        action="reranking",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=tokens_in + tokens_out,
        cost_usd=cost_gemini_flash(tokens_in, tokens_out),
        latency_ms=ms[0],
    )

    score_map = _parse_scores(response.text.strip(), len(candidates))
    if score_map is None:
        logger.warning("Reranker returned unusable JSON; assigning uniform scores")
        return [(c, 5.0) for c in candidates], usage

    scored = [(candidates[i], score_map.get(i, 1.0)) for i in range(len(candidates))]
    return sorted(scored, key=lambda x: x[1], reverse=True), usage
