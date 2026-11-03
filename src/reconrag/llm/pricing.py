"""USD per 1M tokens. Gemini verified 2026-11-01 at ai.google.dev/gemini-api/docs/pricing."""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float | None:
    """None for an unpriced model: an unknown cost must not be reported as zero."""
    price = PRICES.get(model)
    if price is None:
        return None
    return (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000
