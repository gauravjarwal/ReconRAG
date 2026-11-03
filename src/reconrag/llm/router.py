"""Chooses a provider, and falls back only when falling back can help."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from reconrag.errors import AllProvidersFailed, UpstreamRejected, UpstreamUnavailable
from reconrag.llm.base import Provider, Usage
from reconrag.llm.breaker import CircuitBreaker
from reconrag.llm.pricing import cost_usd
from reconrag.observability.metrics import BREAKER_OPEN, LLM_CALLS, LLM_COST, LLM_TOKENS, TTFT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallResult:
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    ttft_ms: float
    total_ms: float


class LLMRouter:
    def __init__(self, providers: Sequence[Provider], breakers: dict[str, CircuitBreaker]) -> None:
        self._providers = list(providers)
        self._breakers = breakers

    @property
    def available(self) -> bool:
        return bool(self._providers)

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | CallResult]:
        """Yield text, then one CallResult.

        Once a token has reached the caller a fallback would restart the answer, so a failure
        after that point is raised instead.
        """
        for provider in self._providers:
            breaker = self._breakers[provider.name]
            if not breaker.allow():
                LLM_CALLS.labels(provider.name, "skipped_open").inc()
                continue
            started = time.perf_counter()
            first_token_at: float | None = None
            usage = Usage(0, 0)
            try:
                for event in provider.stream(system, user, max_tokens):
                    if isinstance(event, Usage):
                        usage = event
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                        TTFT.labels(provider.name).observe(first_token_at - started)
                    yield event
            except UpstreamRejected:
                # The provider answered, so it is healthy, and another would reject this too.
                breaker.record_success()
                LLM_CALLS.labels(provider.name, "rejected").inc()
                raise
            except UpstreamUnavailable as exc:
                breaker.record_failure()
                BREAKER_OPEN.labels(provider.name).set(int(breaker.is_open))
                LLM_CALLS.labels(provider.name, "unavailable").inc()
                if first_token_at is not None:
                    raise
                logger.warning("provider unavailable, trying next", extra={"cause": str(exc)})
                continue

            breaker.record_success()
            BREAKER_OPEN.labels(provider.name).set(0)
            LLM_CALLS.labels(provider.name, "ok").inc()
            yield self._finish(provider, usage, started, first_token_at)
            return
        raise AllProvidersFailed("No model provider is available.")

    @staticmethod
    def _finish(
        provider: Provider, usage: Usage, started: float, first_token_at: float | None
    ) -> CallResult:
        cost = cost_usd(provider.model, usage.tokens_in, usage.tokens_out)
        LLM_TOKENS.labels(provider.name, "in").inc(usage.tokens_in)
        LLM_TOKENS.labels(provider.name, "out").inc(usage.tokens_out)
        if cost is not None:
            LLM_COST.labels(provider.name).inc(cost)
        now = time.perf_counter()
        return CallResult(
            provider=provider.name,
            model=provider.model,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_usd=cost,
            ttft_ms=round(((first_token_at or now) - started) * 1000, 2),
            total_ms=round((now - started) * 1000, 2),
        )

    def complete(self, system: str, user: str, max_tokens: int) -> tuple[str, CallResult]:
        parts: list[str] = []
        for event in self.stream(system, user, max_tokens):
            if isinstance(event, CallResult):
                return "".join(parts), event
            parts.append(event)
        raise AssertionError("stream ended without a CallResult")
