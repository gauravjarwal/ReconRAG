import pytest
from google.genai import errors

from evals.harness.llm import (
    Budget,
    BudgetExceeded,
    GeminiClient,
    LlmResponse,
    StubClient,
    cost_usd,
    with_retries,
)


def _api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "x", "status": "x"}})


def test_cost_uses_the_published_price_table():
    assert cost_usd("gemini-3.1-flash-lite", 1_000_000, 1_000_000) == pytest.approx(0.25 + 1.50)


def test_cost_refuses_a_model_with_no_verified_price():
    with pytest.raises(KeyError):
        cost_usd("gemini-unknown", 1, 1)


def test_budget_blocks_a_call_that_would_exceed_the_cap():
    budget = Budget(max_usd=0.01)
    budget.record(0.009)
    with pytest.raises(BudgetExceeded):
        budget.check(next_call_estimate=0.002)


def test_budget_allows_a_call_inside_the_cap():
    Budget(max_usd=0.01).check(next_call_estimate=0.002)


def test_retries_a_rate_limit_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _api_error(429)
        return "ok"

    assert with_retries(flaky, sleep=lambda _: None) == "ok"
    assert len(calls) == 3


def test_does_not_retry_a_bad_request():
    calls = []

    def bad():
        calls.append(1)
        raise _api_error(400)

    with pytest.raises(errors.APIError):
        with_retries(bad, sleep=lambda _: None)
    assert len(calls) == 1


def test_gives_up_after_the_attempt_limit():
    calls = []

    def down():
        calls.append(1)
        raise _api_error(503)

    with pytest.raises(errors.APIError):
        with_retries(down, attempts=3, sleep=lambda _: None)
    assert len(calls) == 3


def test_stub_costs_nothing_and_is_repeatable():
    client = StubClient()
    first = client.generate("sys", "hello there")
    assert isinstance(first, LlmResponse)
    assert first.cost_usd == 0.0
    assert first.text == client.generate("sys", "hello there").text


def test_gemini_client_needs_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        GeminiClient("gemini-3.1-flash-lite", Budget(max_usd=1.0))
