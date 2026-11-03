"""Second-stage ranking. A cross-encoder reads query and passage together, unlike embeddings."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, TypeAdapter, ValidationError

from reconrag.config import Settings
from reconrag.domain import Hit
from reconrag.errors import UpstreamUnavailable
from reconrag.generation.prompts import RERANK_SYSTEM_V4, render_rerank
from reconrag.llm.router import LLMRouter
from reconrag.observability.metrics import RERANK_SHED

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


class CrossEncoderReranker:
    """Inference is CPU-bound, so extra parallel runs only slow each other down and grow ONNX's
    memory arena. Unbounded, 40 concurrent queries got an API container OOM-killed. Each process
    runs a fixed number at once, and a request that can't get a slot in time degrades instead."""

    def __init__(self, settings: Settings) -> None:
        self._name = settings.cross_encoder_model
        self._cache = str(settings.model_cache_dir)
        self._threads = settings.rerank_threads
        self._wait_s = settings.rerank_wait_s
        self._slots = threading.BoundedSemaphore(settings.rerank_concurrency)
        self._model = None

    def _load(self):  # type: ignore[no-untyped-def]  # fastembed ships no usable stubs.
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self._name, cache_dir=self._cache, threads=self._threads)
        return self._model

    def warm(self) -> None:
        list(self._load().rerank("warm up", ["warm up"]))

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        documents = [f"{hit.context}\n{hit.text}" for hit in hits]
        if not self._slots.acquire(timeout=self._wait_s):
            RERANK_SHED.inc()
            raise UpstreamUnavailable("The reranker is at capacity.")
        try:
            scores = list(self._load().rerank(query, documents))
        finally:
            self._slots.release()
        for hit, score in zip(hits, scores, strict=True):
            hit.rerank_score = float(score)
        return sorted(hits, key=lambda hit: hit.rerank_score or 0.0, reverse=True)


class _Score(BaseModel):
    index: int
    score: float


_SCORES = TypeAdapter(list[_Score])


class LlmReranker:
    """Costs money per query. Bad output raises, so the caller degrades instead of guessing."""

    def __init__(self, router: LLMRouter, max_output_tokens: int = 1024) -> None:
        self._router = router
        self._max_tokens = max_output_tokens

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        raw, _ = self._router.complete(
            RERANK_SYSTEM_V4, render_rerank(query, [h.text for h in hits]), self._max_tokens
        )
        start, end = raw.find("["), raw.rfind("]") + 1
        try:
            scores = _SCORES.validate_python(json.loads(raw[start:end]))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise UpstreamUnavailable("Reranker output wasn't valid JSON.") from exc
        by_index = {item.index: item.score for item in scores}
        if not any(0 <= index < len(hits) for index in by_index):
            raise UpstreamUnavailable("Reranker returned no usable scores.")
        for index, hit in enumerate(hits):
            # An unscored passage ranks last, rather than getting a neutral score it didn't earn.
            hit.rerank_score = by_index.get(index, 0.0)
        return sorted(hits, key=lambda hit: hit.rerank_score or 0.0, reverse=True)


def make_reranker(settings: Settings, router: LLMRouter) -> Reranker | None:
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings)
    if settings.reranker == "llm" and router.available:
        return LlmReranker(router)
    return None
