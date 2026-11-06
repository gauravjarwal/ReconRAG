"""The SSE route must always end with a terminal event, even on an unexpected failure."""

from types import SimpleNamespace

from reconrag.api.routes import query_stream
from reconrag.api.schemas import QueryIn
from reconrag.errors import AllProvidersFailed


def _services(stream):
    return SimpleNamespace(answerer=SimpleNamespace(stream=stream))


def _run(stream):
    body = QueryIn(question="What is the deductible?")
    return list(query_stream(body, SimpleNamespace(), _services(stream)))  # type: ignore[arg-type]  # fakes stand in for the real services


def test_unexpected_failure_mid_stream_ends_with_a_generic_error_event(caplog):
    def broken(collection, question, reidentify):
        yield SimpleNamespace(kind="sources", data={"sources": []})
        raise RuntimeError("secret detail: patient John Doe")

    events = _run(broken)
    assert [e.event for e in events] == ["sources", "error"]
    assert events[-1].data == {
        "code": "internal_error",
        "message": "Something went wrong on our side.",
    }
    assert "John Doe" not in str(events[-1].data)
    assert "unhandled error during stream" in caplog.text


def test_known_error_keeps_its_own_code_and_message():
    def down(collection, question, reidentify):
        raise AllProvidersFailed("No model provider is available.")
        yield

    events = _run(down)
    assert events[-1].event == "error"
    assert events[-1].data == {
        "code": "AllProvidersFailed",
        "message": "No model provider is available.",
    }
