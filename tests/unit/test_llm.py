from collections.abc import Iterator

import pytest

from reconrag.errors import AllProvidersFailed, UpstreamRejected, UpstreamUnavailable
from reconrag.llm.base import Usage
from reconrag.llm.breaker import CircuitBreaker
from reconrag.llm.pricing import cost_usd
from reconrag.llm.router import CallResult, LLMRouter


class Scripted:
    """Yields its script in order and raises any exception found in it."""

    def __init__(self, name: str, script: list[object]) -> None:
        self.name = name
        self.model = "claude-haiku-5-5"
        self.script = script
        self.calls = 0

    def stream(self, system: str, user: str, max_tokens: int) -> Iterator[str | Usage]:
        self.calls += 1
        for step in self.script:
            if isinstance(step, Exception):
                raise step
            yield step  # type: ignore[misc]


def router(*providers: Scripted, threshold: int = 3) -> LLMRouter:
    breakers = {p.name: CircuitBreaker(threshold, 30.0) for p in providers}
    return LLMRouter(list(providers), breakers)


def test_cost_uses_listed_prices():
    assert cost_usd("gemini-3.1-flash-lite", 1_000_000, 1_000_000) == pytest.approx(1.75)


def test_unpriced_model_has_unknown_cost_not_zero():
    assert cost_usd("some-new-model", 1000, 1000) is None


def test_stream_ends_with_call_result():
    provider = Scripted("a", ["he", "llo", Usage(10, 2)])
    events = list(router(provider).stream("s", "u", 10))
    assert events[:2] == ["he", "llo"]
    assert isinstance(events[-1], CallResult)
    assert (events[-1].tokens_in, events[-1].tokens_out) == (10, 2)


def test_falls_back_when_first_provider_is_unavailable():
    first = Scripted("a", [UpstreamUnavailable("503")])
    second = Scripted("b", ["ok", Usage(1, 1)])
    text, result = router(first, second).complete("s", "u", 10)
    assert text == "ok"
    assert result.provider == "b"


def test_rejected_request_does_not_fall_back():
    first = Scripted("a", [UpstreamRejected("400")])
    second = Scripted("b", ["ok", Usage(1, 1)])
    with pytest.raises(UpstreamRejected):
        router(first, second).complete("s", "u", 10)
    assert second.calls == 0


def test_failure_after_first_token_raises_instead_of_restarting():
    first = Scripted("a", ["partial", UpstreamUnavailable("dropped")])
    second = Scripted("b", ["ok", Usage(1, 1)])
    stream = router(first, second).stream("s", "u", 10)
    assert next(stream) == "partial"
    with pytest.raises(UpstreamUnavailable):
        next(stream)
    assert second.calls == 0


def test_all_providers_down():
    with pytest.raises(AllProvidersFailed):
        router(Scripted("a", [UpstreamUnavailable("x")])).complete("s", "u", 10)


def test_open_breaker_skips_provider():
    failing = Scripted("a", [UpstreamUnavailable("x")])
    backup = Scripted("b", ["ok", Usage(1, 1)])
    routed = router(failing, backup, threshold=2)
    routed.complete("s", "u", 10)
    routed.complete("s", "u", 10)
    routed.complete("s", "u", 10)
    assert failing.calls == 2
    assert backup.calls == 3


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_at_threshold_and_ignores_older_success():
    breaker = CircuitBreaker(2, 10.0, Clock())
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.allow()
    breaker.record_failure()
    assert not breaker.allow()


def test_breaker_allows_one_trial_after_cooldown():
    clock = Clock()
    breaker = CircuitBreaker(1, 10.0, clock)
    breaker.record_failure()
    assert not breaker.allow()
    clock.now = 11
    assert breaker.allow()
    assert not breaker.allow(), "only one trial at a time"


def test_failed_trial_reopens_and_successful_trial_closes():
    clock = Clock()
    breaker = CircuitBreaker(1, 10.0, clock)
    breaker.record_failure()
    clock.now = 11
    breaker.allow()
    breaker.record_failure()
    assert not breaker.allow()
    clock.now = 22
    breaker.allow()
    breaker.record_success()
    assert breaker.allow()
    assert not breaker.is_open
