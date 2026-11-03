"""Every model call in the evals goes through here, so each one is logged, priced and capped.

Sources:
- SDK: https://github.com/googleapis/python-genai (HttpOptions.timeout is in milliseconds).
- Prices: https://ai.google.dev/gemini-api/docs/pricing, read on 2026-11-01. Output prices include
  thinking tokens, so thinking tokens are counted as output.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

logger = logging.getLogger("evals.llm")

# USD per 1M tokens: (input, output).
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}
DEFAULT_MODEL = "gemini-3.1-flash-lite"
TIMEOUT_MS = 30_000
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


class BudgetExceeded(RuntimeError):
    """The next call would pass the spend cap."""


class LlmResponse(BaseModel):
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float


class LlmClient(Protocol):
    model: str

    def generate(self, system: str, user: str, json_output: bool = False) -> LlmResponse: ...


@dataclass
class Budget:
    """A hard cap. The run stops before spending past it, not after."""

    max_usd: float
    spent_usd: float = 0.0

    def check(self, next_call_estimate: float) -> None:
        if self.spent_usd + next_call_estimate > self.max_usd:
            raise BudgetExceeded(
                f"Spent ${self.spent_usd:.4f}; next call could pass the ${self.max_usd:.2f} cap."
            )

    def record(self, amount: float) -> None:
        self.spent_usd += amount


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES_PER_MTOK[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in RETRYABLE_CODES
    return isinstance(exc, httpx.TransportError)


def with_retries[T](
    call: Callable[[], T],
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry transient failures with exponential backoff and full jitter."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except (errors.APIError, httpx.TransportError) as exc:
            if attempt == attempts or not _retryable(exc):
                raise
            delay = random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1)))
            logger.warning("Retry %d after %s", attempt, type(exc).__name__)
            sleep(delay)
    raise AssertionError("unreachable")


class GeminiClient:
    def __init__(self, model: str, budget: Budget, max_output_tokens: int = 1024) -> None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("Set GEMINI_API_KEY to run live generation evals.")
        self.model = model
        self.budget = budget
        self.max_output_tokens = max_output_tokens
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS)
        )

    def generate(self, system: str, user: str, json_output: bool = False) -> LlmResponse:
        # Worst case for this call: 1 token per 3 characters in, plus the full output allowance.
        estimate = cost_usd(self.model, (len(system) + len(user)) // 3, self.max_output_tokens)
        self.budget.check(estimate)
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json" if json_output else None,
        )
        started = time.perf_counter()
        response = with_retries(
            lambda: self._client.models.generate_content(
                model=self.model, contents=user, config=config
            )
        )
        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.usage_metadata
        tokens_in = (usage.prompt_token_count or 0) if usage else 0
        tokens_out = 0
        if usage:
            tokens_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        cost = cost_usd(self.model, tokens_in, tokens_out)
        self.budget.record(cost)
        logger.info(
            "llm model=%s tokens_in=%d tokens_out=%d latency_ms=%.0f cost_usd=%.6f",
            self.model,
            tokens_in,
            tokens_out,
            latency_ms,
            cost,
        )
        return LlmResponse(
            text=response.text or "",
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
        )


class StubClient:
    """No network and no cost. It proves the harness runs, and says nothing about quality."""

    model = "stub"

    def generate(self, system: str, user: str, json_output: bool = False) -> LlmResponse:
        if json_output:
            text = '{"faithful": true, "reason": "stub"}'
        elif "<context" in user:
            first = user.split(">", 1)[1].split("</context>")[0].strip()
            text = f"{first[:200]} [Source: S1 | Chunk: 0]"
        else:
            text = "The context does not contain this information."
        return LlmResponse(
            text=text,
            model=self.model,
            tokens_in=len(user) // 4,
            tokens_out=len(text) // 4,
            latency_ms=0.0,
            cost_usd=0.0,
        )
