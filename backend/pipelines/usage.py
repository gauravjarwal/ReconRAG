from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

# ---------------------------------------------------------------------------
# Pricing constants (USD per 1 M tokens unless noted)
# ---------------------------------------------------------------------------

# Claude Haiku 4.5
CLAUDE_SONNET_INPUT_PER_MTK: float = 1.00
CLAUDE_SONNET_OUTPUT_PER_MTK: float = 5.00

# Gemini 3.1 Flash Lite (reranker + generation fallback)
GEMINI_FLASH_INPUT_PER_MTK: float = 0.075
GEMINI_FLASH_OUTPUT_PER_MTK: float = 0.30

# Gemini gemini-embedding-001: $0.00001 per 1 000 characters
# Approximating 4 chars/token → ~$0.04 per 1 M tokens
GEMINI_EMBEDDING_PER_1K_CHARS: float = 0.00001


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ActionUsage:
    """Token usage and cost for a single pipeline action."""
    action: str
    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    pct_of_total_tokens: float = 0.0   # filled by aggregate_usage()


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def cost_claude(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * CLAUDE_SONNET_INPUT_PER_MTK + tokens_out * CLAUDE_SONNET_OUTPUT_PER_MTK) / 1_000_000


def cost_gemini_flash(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * GEMINI_FLASH_INPUT_PER_MTK + tokens_out * GEMINI_FLASH_OUTPUT_PER_MTK) / 1_000_000


def cost_gemini_embedding(total_chars: int) -> float:
    """Character-based pricing for Gemini embedding API."""
    return (total_chars / 1_000) * GEMINI_EMBEDDING_PER_1K_CHARS


def estimate_tokens(text: str) -> int:
    """Rough token estimate when the API does not return a count (1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Latency context manager
# ---------------------------------------------------------------------------

@contextmanager
def timed() -> Generator[list[float], None, None]:
    """
    Usage::

        with timed() as ms:
            do_work()
        elapsed = ms[0]
    """
    bucket: list[float] = [0.0]
    t0 = time.perf_counter()
    try:
        yield bucket
    finally:
        bucket[0] = (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_usage(actions: list[ActionUsage]) -> dict:
    """
    Compute totals and fill pct_of_total_tokens in-place.

    Returns a dict ready for UsageSummary construction.
    """
    total_tokens = sum(a.total_tokens for a in actions)
    total_cost = sum(a.cost_usd for a in actions)
    total_latency = sum(a.latency_ms for a in actions)

    for a in actions:
        a.pct_of_total_tokens = round(a.total_tokens / total_tokens * 100, 2) if total_tokens else 0.0

    return {
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 8),
        "total_latency_ms": round(total_latency, 2),
        "breakdown": [
            {
                "action": a.action,
                "tokens_in": a.tokens_in,
                "tokens_out": a.tokens_out,
                "total_tokens": a.total_tokens,
                "cost_usd": round(a.cost_usd, 8),
                "latency_ms": round(a.latency_ms, 2),
                "pct_of_total_tokens": a.pct_of_total_tokens,
            }
            for a in actions
        ],
    }
