"""A cross-encoder on CPU is the query path's bottleneck, so its concurrency must be bounded."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from reconrag.config import Settings
from reconrag.errors import UpstreamUnavailable
from reconrag.retrieval.rerank import CrossEncoderReranker


class SlowModel:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.running = 0
        self.peak = 0
        self.lock = threading.Lock()

    def rerank(self, query, documents):  # type: ignore[no-untyped-def]
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        time.sleep(self.seconds)
        with self.lock:
            self.running -= 1
        return [1.0 for _ in documents]


def reranker(model: SlowModel, slots: int, wait: float) -> CrossEncoderReranker:
    settings = Settings(_env_file=None, rerank_concurrency=slots, rerank_wait_s=wait)  # type: ignore[call-arg]
    instance = CrossEncoderReranker(settings)
    instance._model = model
    return instance


def hits() -> list:
    return [SimpleNamespace(context="", text="x", rerank_score=None)]


def test_no_more_than_the_configured_number_run_at_once():
    model = SlowModel(0.02)
    instance = reranker(model, slots=2, wait=5)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: instance.rerank("q", hits()), range(16)))
    assert model.peak == 2


def test_a_request_gives_up_instead_of_queueing_forever():
    model = SlowModel(0.3)
    instance = reranker(model, slots=1, wait=0.01)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(instance.rerank, "q", hits())
        time.sleep(0.05)
        with pytest.raises(UpstreamUnavailable):
            instance.rerank("q", hits())
        first.result()


def test_a_failed_inference_gives_its_slot_back():
    model = SlowModel(0)
    instance = reranker(model, slots=1, wait=0.01)
    model.rerank = lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        instance.rerank("q", hits())
    model.rerank = lambda query, documents: [1.0 for _ in documents]  # type: ignore[method-assign]
    assert instance.rerank("q", hits())
